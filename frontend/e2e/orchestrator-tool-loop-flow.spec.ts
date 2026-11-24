import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { promisify } from "node:util";

import { expect, test, type Page } from "@playwright/test";

const execute = promisify(execFile);
const workloadPrefix = "Check workload and rebalance if needed for ";

type ConversationSnapshot = {
  messages: Array<{
    role: "USER" | "ASSISTANT";
    content_blocks: Array<{ kind: string; code?: string; text?: string }>;
  }>;
};

async function signIn(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill("manager@example.test");
  await page.getByLabel("Mật khẩu").fill("WorkDemo123!");
  await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
  await expect(page.getByRole("button", { name: "Đăng xuất" })).toBeVisible();
  await page.getByRole("button", { name: "en", exact: true }).click();
  await expect(page.getByLabel("Message the AI Assistant")).toBeVisible();
}

async function send(page: Page, text: string) {
  const accepted = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      /\/api\/v1\/ai\/conversations\/[^/]+\/messages$/.test(response.url()),
  );
  await page.getByLabel("Message the AI Assistant").fill(text);
  await page.getByRole("button", { name: "Send", exact: true }).click();
  expect((await accepted).status()).toBe(202);
  await expect(page).toHaveURL(/[?&]conversation=[0-9a-f-]+(?:&|$)/);
}

async function snapshot(page: Page): Promise<ConversationSnapshot> {
  const id = new URL(page.url()).searchParams.get("conversation");
  expect(id).toBeTruthy();
  const response = await page.request.get(`/api/v1/ai/conversations/${id}`);
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function post<T>(page: Page, path: string, data: object): Promise<T> {
  const response = await page.request.post(`/api/v1${path}`, {
    headers: { "Idempotency-Key": randomUUID() },
    data,
  });
  expect(response.status(), await response.text()).toBe(201);
  return response.json();
}

async function seedWorkload(page: Page, effort: 32 | 48) {
  const members = await page.request.get("/api/v1/members?is_active=true&page_size=100");
  expect(members.ok()).toBeTruthy();
  const { items } = (await members.json()) as {
    items: Array<{ display_name: string; membership_id: string }>;
  };
  const member = items.find((item) => item.display_name === "Demo Employee");
  expect(member).toBeDefined();
  const project = `Browser launch ${randomUUID()}`;
  const row = await post<{ id: string }>(page, "/projects", { name: project });
  // Run each fixture in an independent week so another case cannot change its workload.
  const monday = new Date("2040-01-02T00:00:00Z");
  const discriminator = Number.parseInt(project.slice(-8), 16) % 1500;
  monday.setUTCDate(monday.getUTCDate() + (discriminator * 2 + Number(effort === 48)) * 7);
  const sunday = new Date(monday);
  sunday.setUTCDate(sunday.getUTCDate() + 6);
  const start = monday.toISOString().slice(0, 10);
  const end = sunday.toISOString().slice(0, 10);
  const week = await post<{ id: string }>(page, `/projects/${row.id}/weeks`, {
    week_number: 1,
    start_date: start,
    end_date: end,
    objective: "Deliver launch work",
  });
  await post(page, "/capacity", {
    membership_id: member!.membership_id,
    kind: "OVERRIDE",
    week_start: start,
    hours: 40,
  });
  const task = await post<{ id: string }>(page, "/tasks", {
    project_id: row.id,
    project_week_id: week.id,
    title: "Prepare launch deliverables",
    estimated_effort_hours: effort,
    due_date: end,
  });
  // Seed assignment as test data. The workflow under test never approves a team
  // or mutates any allocation.
  await execute("uv", [
    "run", "--locked", "python", "-m", "tests.orchestrator_e2e_data",
    process.env.ORCHESTRATOR_E2E_ORGANIZATION_ID!, task.id, member!.membership_id,
  ], {
    cwd: resolve(process.cwd(), "../backend"),
    env: { ...process.env, APP_DATABASE_URL: process.env.ORCHESTRATOR_E2E_DATABASE_URL },
    timeout: 20_000,
  });
  return project;
}

type PlanningRun = { id: string; input_goal_text: string; status: string };

async function relatedRuns(page: Page, project: string) {
  const response = await page.request.get("/api/v1/workflow-runs?limit=100");
  expect(response.ok()).toBeTruthy();
  const body = (await response.json()) as { items: PlanningRun[] };
  return body.items.filter((run) => run.input_goal_text.includes(project));
}

test("candidate greeting persists across a reload", async ({ page }) => {
  await signIn(page);
  await send(page, "Hello, Orchestrator");
  const answer = "Hello! How can I help with your work?";
  await expect(page.getByText(answer, { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText(answer, { exact: true })).toHaveCount(1);
  const replies = (await snapshot(page)).messages.filter((message) => message.role === "ASSISTANT");
  expect(replies.flatMap((message) => message.content_blocks)).toEqual([
    { kind: "text", text: answer },
  ]);
});

test("verified 32/40 workload needs no planning follow-up", async ({ page }) => {
  await signIn(page);
  const project = await seedWorkload(page, 32);
  await send(page, `${workloadPrefix}${project}`);
  await expect(page.getByText("Workload is 32/40 hours. No rebalance is needed.", { exact: true })).toBeVisible();
  expect(await relatedRuns(page, project)).toHaveLength(0);
});

test("verified 48/40 workload leads to an unapplied approval-gated proposal", async ({ page }) => {
  await signIn(page);
  const project = await seedWorkload(page, 48);
  await send(page, `${workloadPrefix}${project}`);

  await expect.poll(async () => (await relatedRuns(page, project)).map((run) => run.status)).toEqual([
    "WAITING_FOR_DECISION",
  ]);
  const [run] = await relatedRuns(page, project);
  expect(run.input_goal_text).toContain("48/40 hours");
  const response = await page.request.get(`/api/v1/workflow-runs/${run.id}`);
  expect(response.ok()).toBeTruthy();
  const saved = await response.json();
  expect(saved.current_proposal).toMatchObject({
    status: "READY_FOR_DECISION",
    version: 1,
    approval_id: expect.any(String),
    proposal_id: expect.any(String),
  });
  expect(saved.allowed_actions).toContain("DECIDE_APPROVAL");
  expect(saved.project_id).toBeNull();
  await expect(page.getByText("A proposal is ready and awaiting Manager approval.", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText("A proposal is ready and awaiting Manager approval.", { exact: true })).toHaveCount(1);
  const cards = (await snapshot(page)).messages.flatMap((message) =>
    message.content_blocks.filter((block) => block.kind === "proposal"),
  );
  expect(cards).toEqual([
    expect.objectContaining({
      workflow_run_id: run.id,
      proposal_id: saved.current_proposal.proposal_id,
      proposal_version: saved.current_proposal.version,
      approval_id: saved.current_proposal.approval_id,
    }),
  ]);
  expect(await relatedRuns(page, project)).toHaveLength(1);
});

test("a failed model shows a recoverable manual action", async ({ page }) => {
  await signIn(page);
  await send(page, "Exercise Orchestrator fallback");
  await expect(page.getByRole("heading", { name: "Unable to complete request", exact: true })).toBeVisible();
  await page.reload();
  await expect(page.locator(".assistant-safe-error")).toHaveCount(1);
  expect((await snapshot(page)).messages.flatMap((message) => message.content_blocks)).toContainEqual(
    expect.objectContaining({ kind: "safe_error", code: "MODEL_UNAVAILABLE" }),
  );
  await page.getByRole("button", { name: "Continue manually", exact: true }).click();
  await expect(page.getByRole("button", { name: "Create project", exact: true })).toBeVisible();
});
