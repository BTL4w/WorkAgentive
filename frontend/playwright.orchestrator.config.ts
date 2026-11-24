import { defineConfig, devices } from "@playwright/test";

const databaseUrl = process.env.ORCHESTRATOR_E2E_DATABASE_URL;
const organization = process.env.ORCHESTRATOR_E2E_ORGANIZATION_ID;
if (!databaseUrl || !organization) {
  throw new Error("Set the dedicated Orchestrator E2E database and organization");
}
const database = new URL(databaseUrl);
if (
  database.protocol !== "postgresql+psycopg:" ||
  !["localhost", "127.0.0.1"].includes(database.hostname) ||
  database.pathname !== "/work_management_orchestrator_e2e" ||
  database.search ||
  database.hash ||
  !/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(organization)
) {
  throw new Error("Orchestrator E2E requires the dedicated local synthetic database");
}

const api = "http://127.0.0.1:8124";
const browser = "http://127.0.0.1:3124";
const env = {
  APP_DATABASE_URL: databaseUrl,
  APP_ENVIRONMENT: "local",
  APP_AI_PROVIDER: "mock",
  APP_AI_MODEL: "mock:orchestrator-e2e-v1",
  APP_ORCHESTRATOR_DRIVER: "tool_loop",
  APP_ORCHESTRATOR_TOOL_LOOP_ORGANIZATIONS: JSON.stringify([organization]),
  APP_WORKER_ORGANIZATION_IDS: JSON.stringify([organization]),
  APP_PROGRESSIVE_PLANNING_ENABLED: "false",
  APP_LANGSMITH_TRACING: "false",
  LANGSMITH_TRACING: "false",
  APP_RETRIEVAL_ENABLED: "false",
};

export default defineConfig({
  testDir: "./e2e",
  testMatch: "orchestrator-tool-loop-flow.spec.ts",
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 25_000 },
  retries: 0,
  reporter: "list",
  use: { baseURL: browser, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: devices["Desktop Chrome"] }],
  webServer: [
    {
      command: "cd ../backend && uv run --locked fastapi run app/main.py --host 127.0.0.1 --port 8124 --no-proxy-headers",
      env,
      url: `${api}/openapi.json`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "cd ../backend && uv run --locked python -m tests.orchestrator_e2e_worker",
      env,
      wait: { stderr: /Worker .* started/ },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "corepack pnpm@10 start --hostname 127.0.0.1 --port 3124",
      env: { API_ORIGIN: api, NEXT_DIST_DIR: ".next-orchestrator-e2e" },
      url: `${browser}/login`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
