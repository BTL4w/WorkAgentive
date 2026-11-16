# Bổ sung evaluation: chứng minh tool loop

Ngày 2026-11-16. Giữ nguyên corpus 60 cases của Task 0. Các cases đó là authored
acceptance scenarios, không phải kết quả đo routing hoặc native runtime.

## Phạm vi ngay lúc này

Chỉ tập trung Orchestrator nhận tool result rồi chọn bước tiếp. Không sửa
Specialist internals hoặc thêm business capability. Các luồng end-to-end lớn
(plan → team → assignment, execution sau approval, requirement changes, crash
giữa nhiều business commits, hai Manager sửa/duyệt, Daily Update → Reporting,
Risk → replan, team thiếu skill, partial success nhiều Agent và project-level
authorization) vẫn quan trọng nhưng chưa là tiêu chí bắt buộc của đợt này.
Các hard guards về quyền/tenant/approval không bị trì hoãn; chỉ hoãn bài kiểm
thử tích hợp nghiệp vụ rộng. Không xóa những cases đã có trong corpus cũ.

## Phân tầng chứng cứ

| Lớp | Chứng minh gì | Không chứng minh gì |
| --- | --- | --- |
| Corpus validation | Cases/fixtures nhất quán | Agent thật chọn đúng tool |
| Authored trace checker | Phát hiện thiếu observation, sai causal ordering, dừng sớm, nhánh sai hoặc effect trái phép trong trace được cung cấp | Runtime đã tạo trace này |
| Runtime replay | Chạy driver/pipeline thật với scripted model + Specialist boundaries stub; capture input mỗi model step, calls/results và session events thực | Hosted model tự chọn đúng hành động |
| Live model eval | Configured model chọn specialist/next action hợp lý theo observations | Không thay RLS/transaction/approval integration tests |

Hiện bổ sung lớp authored trace checker và fixtures; runtime replay phải nối
vào driver thật ở Tasks 1/10/14. Không viết một vòng lặp giả chỉ để tự pass test.
Không lấy trường evidence_origin từ input làm bằng chứng tin cậy cho provenance;
harness thực tế phải sở hữu capture/measurement và không load authored expected
events thành observed events. Hosted vẫn opt-in, credential-gated.

## Hai nhánh workload bắt buộc

Cùng prompt trong mỗi ngôn ngữ: kiểm tra workload rồi đề xuất nếu quá tải.
Tool boundary đầu là `delegate_agent` → Assignment `assignment.analyze_workload`:
đây là ownership đang có trong repo, không thêm workload tool cho Work Intelligence.

- Capacity 40h, allocated 32h: model nhận snapshot 0.8, trả lời có evidence;
  không gọi Planning chỉ để làm đủ checklist.
- Capacity 40h, allocated 48h: model nhận snapshot 1.2, chọn Planning/Assignment
  phù hợp để chuẩn bị phương án. Fixture minh họa dùng `planning.revise`;
  proposal có reference/version, dừng chờ người duyệt, không tự apply.

`orchestrator_tool_loop_traces.jsonl` có 4 bilingual fixtures, label rõ
`AUTHORED_NOT_RUNTIME_CAPTURE`. Snapshot dùng existing `ProjectWorkloadSnapshot`.
Model inputs chứa tool result cụ thể, không chỉ nhãn "overloaded=true". Events
là normalized evaluation vocabulary, không tuyên bố là schema DSH hoặc public
event API của WorkCrew; future adapter map actual runtime events sang vocabulary.

## Runtime replay acceptance phải thực hiện sau

1. Scripted model phát native delegate call; real pipeline invoke đúng boundary.
2. Result của backend/stub do handler tạo, không chép expected trace vào history.
3. Captured request N+1 chứa đúng call ID và result từ request N; thiếu/sai/mất
   result làm test fail. Giữ state/session mới cho mỗi test, không contamination.
4. Cùng mục tiêu nhưng đổi workload response phải kiểm tra hai next decisions.
   Script chỉ được tiếp tục khi observed input đúng; không assert mỗi final label.
5. Assert consumed model steps (`assert_complete`), no extra calls, settled tool
   calls, đúng proposal/version/human wait và zero pre-approval business effects.
6. Thêm mutation tests cố tình bỏ result, duplicate call, response trước request,
   kết thúc sớm, bỏ nhánh Planning khi overload và planning thừa khi bình thường.
7. Scripted replay chứng minh plumbing/control flow; chỉ live eval được đo actual
   model selection. Không snapshot toàn final prose, timestamps hoặc UUID ngẫu nhiên.

Fixture minh họa không khóa mọi valid execution path. Runtime suite cho phép
đường chọn Specialist khác nếu đúng capability, evidence và outcome; protocol
causality/identity/human gates vẫn phải chính xác.

## Nguồn official đã đối chiếu

- [DSH testing policy tại 39163dd3](https://github.com/deepseek-ai/deepseek-harness/blob/39163dd30f703a79079e2d2446545cff3c453738/docs/testing.md): policy yêu cầu recorded-session scenarios cho thay đổi model/protocol/UX đáng kể; keyless replay khác real-API E2E. WorkCrew học replay qua assembled runtime, không import nguyên testkit TypeScript.
- [OpenAI Agents SDK Testing](https://openai.github.io/openai-agents-python/testing/): ScriptedModel, inspect model calls và assert_complete hỗ trợ kiểm tra tool workflow/unfinished steps. Áp dụng pattern vào gateway mock hiện có, không thêm SDK dependency.
- [LangGraph testing](https://docs.langchain.com/oss/python/langgraph/test): fresh graph/checkpointer mỗi test, full graph và partial-state tests.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence): state/checkpoints phục vụ continuation/recovery, không thay application idempotency. Chưa triển khai broad crash/business-commit suites trong bổ sung này.

Tài liệu living docs kiểm tra ngày 2026-11-16. AutoGen/CrewAI/Pydantic AI là
nguồn tham khảo bổ sung do user cung cấp; chưa xác minh chi tiết API trong đợt
này và không dùng chúng làm dependency hoặc nguồn cấp quyền cho WorkCrew.

## Kết quả kiểm chứng phần bổ sung

- RED: missing verifier module; các tests business-effect flag và proposal wait
  version từng fail đúng assertion trước khi bổ sung validation.
- GREEN: 7 tests trace checker, existing corpus test giữ nguyên.
- Ruff lint/format PASS; pyright 0 errors; isolated AI unit suite **548 passed,
  3 deselected**. Không chạy hosted hoặc DB integration trong đợt này.
- Full core evaluation không được tuyên bố pass; baseline PostgreSQL NOT_RUN
  đã ghi trước đó vẫn là giới hạn chưa được giải quyết.
- Original corpus byte-identical với commit trước; không đổi Agent manifests,
  Specialist harnesses, model configuration hoặc runtime driver.

## Amendment tiếp theo — làm cụ thể ngay trong corpus gốc

Theo phản hồi người dùng, "giữ nguyên" được sửa cách áp dụng: giữ 60 IDs, 30
scenario intents và prompt cũ trong `original_prompt`, không giữ nguyên các
prompt chung chung. File JSONL gốc giờ chứa prompt mới, `initial_context`,
`environment`, `tool_fixtures`, constraints và label MODEL_DECISION/RUNTIME_CONTRACT.

Dữ liệu synthetic về SAFE-12: đào tạo 20 người, hai nhóm, kho Bình Tân, deadline
18/12/2026, giữ ca sáng; workload tuần 30/11–06/12 có nguồn task cộng đúng 32h
hoặc 48h trên capacity 40h. Hai nhánh dùng cùng prompt, đổi tool result, yêu cầu
observe trước next decision và không auto-apply. Case tìm Project có candidates
Bình Tân/Hải Phòng; progress có 6/10 checklist items, runtime crash có receipt/
checkpoint/failpoint tách khỏi prompt tự nhiên.

`environment` là ground-truth/backend seed, KHÔNG gửi toàn bộ vào model.
`initial_context` là bootstrap; `tool_fixtures` chỉ được trả khi đúng boundary
được gọi. Keys như assignment.analyze_workload là capability fixtures cho
`delegate_agent`, không tự trở thành model-facing tools hoặc mở rộng Agent
allowlists. Fixture contract ghi EVALUATION_BOUNDARY_NOT_API_WIRE để tránh coi
các fields minh họa này là current Specialist wire schema.

Skill-version case bỏ DRAFT_READY khỏi acceptable outcomes vì yêu cầu chỉ tìm
context, không yêu cầu tạo kế hoạch. Các outcome labels khác không tự trở thành
runtime evidence. Tất cả 60 cases ghi AUTHORED_NOT_EXECUTED; runner/model-quality
measurement vẫn chưa có. Bộ trace riêng không thay thế nội dung corpus gốc.


## Correction — yêu cầu người dùng phải có nghĩa khi đọc

Bỏ mã giả `SAFE-12` khỏi prompt và dữ liệu hiện hành của cả 60 cases. Mỗi
request độc lập nói rõ công việc, ví dụ đào tạo an toàn lao động, kiểm tra thiết
bị bảo hộ hoặc tổ chức buổi thực hành sơ cứu. Các câu nối tiếp có
`initial_context.conversation_history` Việt/Anh tương ứng; câu hỏi đang chờ có
run/question/version trong `environment.pending_inputs`. Case UI có payload
`verified_action` riêng, không giả làm yêu cầu chat phải route qua model.
Tên kho, người và công ty đều là dữ liệu synthetic, không phải thông tin của
người dùng. Giữ nguyên IDs, scenario intents, original_prompt và safety bằng 0.

Case thiếu mục tiêu giờ là:

> Tháng 12 tôi muốn tổ chức một buổi đào tạo cho nhân viên kho Bình Tân. Bạn giúp tôi lên kế hoạch nhé.

Việc cần lập kế hoạch là tổ chức đào tạo; phần còn thiếu là chủ đề/kết quả mong
muốn, số người và thời điểm cụ thể. Backend seed không còn giấu sẵn một dự án
đào tạo an toàn có đầy đủ mục tiêu rồi bắt model hỏi lại: search không có
candidate, không có read_project_context. Chấp nhận ASK_USER để làm rõ nhu cầu,
không tự bịa chủ đề đào tạo an toàn. Case optional_budget có đủ mục tiêu và cho
phép ước tính chi phí nên không chặn chỉ vì ngân sách chưa chốt.

Case context mới có snapshot chứa nội dung đã đọc, source/version, ràng buộc và
lịch hai nhóm, thay vì chỉ một reference. Quyền bị thu hồi trả FORBIDDEN; nguồn
hết hạn không trả payload cũ. Conversation/snapshot không cấp quyền: runtime
vẫn phải kiểm tra quyền thật ở mỗi boundary khi được triển khai.

Cả bốn authored traces cũng dùng câu hỏi có tên dự án, người và tuần cụ thể;
identity bindings nối Lan/project/week với các ID trong snapshot. Capacity và
allocated hours chỉ xuất hiện trong tool result. Objective giao Assignment chỉ
đọc workload; quyền quyết định gọi Planning tiếp vẫn ở Orchestrator.

Kiểm chứng: năm assertions mới/sửa đã fail trên corpus cũ; sau sửa, 13 tests
corpus/trace PASS. AI suite 553 passed, 3 deselected; Ruff lint/format PASS và
pyright 0 errors. Rà diff xác nhận 60/60 prompt thay đổi, không mất ID hoặc
original_prompt. Đây là kiểm chứng cấu trúc và tính nhất quán fixture; chưa đo
hosted model, chưa chứng minh native driver chạy và chưa đóng full AI gate.


## Task 1 — Native protocol và v2 state contracts

Scope: contracts/mock/test fakes only. Added provider-neutral ToolSchema,
ToolCall, ToolItem, ToolModelRequest/Response and NativeToolGateway alongside
unchanged structured v1 API. Valid responses carry one tool_call or one typed
final; commentary alone, multiple calls, tool+final, mismatched finish, orphan/
duplicate/unsettled history and alias collisions fail validation. Wire alias
`automation__preview` maps from canonical `automation.preview@1` and is not an
alternate audit identity. Registered input contracts validate mock arguments;
JSON schema alone is declaration metadata, not an authorization check.

RunScope/RunLease, limits/usage, version pins, expiring payload/wait references,
ToolLoopState/Checkpoint, AdmittedToolSet and typed backend outcomes are shared
interfaces for later tasks. Checkpoint has no raw prompt/context/approval/role
fields. Scope checks reject payload/wait references from another scope; actual
RLS, leases, budget reservation and policy enforcement remain later work.
PublishedResponse preserves the four existing statuses. SuspendedWorkflow has
WORKFLOW_RESULT wait/activity/cursor and cannot claim a terminal public status.

MockModelGateway.generate_step consumes independent native scripts by invocation
key, captures deep copies of requests and exposes assert_complete. Legacy
sequence_fixtures and native tool_sequence_fixtures have separate queues; fixtures
and captures cannot be changed by later caller mutation. Fake clock/checkpoint
store/backend live only in tests, not production runtime. Tests manually supply
observations to gateway requests: no claim that a real driver generated them.

Research: re-inspected [DSH agent.ts at 39163dd3](https://github.com/deepseek-ai/deepseek-harness/blob/39163dd30f703a79079e2d2446545cff3c453738/packages/core/agent-loop/src/agent.ts),
2026-11-16. Adopt provider-neutral history and step-scoped admitted tools; adapt
to WorkCrew one-call steps and tenant-owned contracts. No SDK/framework import,
model change or DSH child/backend activation. A guessed adjacent types.ts URL
returned 404 and was not used as evidence.

Verification:

- Initial targeted RED: missing tool_contracts module. Added fake boundary test
  also exposed BaseModel serialization losing output fields; SerializeAsAny now
  preserves registered typed backend output/canonical arguments.
- Targeted native + existing gateway: 31 passed (23 new, 8 existing).
- make ai: Ruff lint/format PASS, pyright 0 errors, unit suite 576 passed /
  3 live-provider tests deselected. Core gate exits nonzero: progressive planning
  has 14 PASS and 6 PostgreSQL cases NOT_RUN, same baseline gap; other six gates
  pass. Full AI gate is not claimed green; no DB reset or hosted invocation.
- make -C ai eval-retrieve: PASS, required_total 28, safety violations 0,
  hosted_status NOT_RUN. Run separately because make ai stopped at core eval.
- Diff reviewed: only Task 1 files and this verification note staged. Production
  registration/admission, specialist implementations and v1 contracts unchanged.


## Task 2 — Hosted native tool adapter (offline verified)

Added `model_gateway/openai_tools.py`: OpenAINativeToolGateway implements
NativeToolGateway.generate_step with constructor-injected model name/key/factory.
Shared private_provider_request in the existing adapter suppresses SDK dumps/
tracing; legacy structured generation retains its existing transport/behavior.
No model configuration, title model, dependency, production admission or
Specialist changes. No hosted API call was made.

Native transport is Chat Completions through the installed ChatOpenAI adapter:
full typed history, assistant tool_calls IDs and ToolMessage.tool_call_id are
rebuilt each request, store=False, max_retries=0. No previous_response_id or
provider-hosted session/opaque reasoning is retained. This is distinct from the
unchanged Responses path of structured v1. A configured model must support this
transport plus strict functions and final JSON schema; otherwise return typed
unavailable, without changing model or weakening the request.

Strict compiler preserves supported validation keywords, removes defaults and
requires every declared property (nullable types stay nullable). Open maps,
unknown schema keywords and non-object roots fail before provider creation.
Tools require the registered Pydantic input contract. Unsupported schemas are
not silently loosened or dropped. Binding uses strict=True and
parallel_tool_calls=False with strict final response_format. Local validation
rejects unknown/reused calls, malformed arguments/final, multiple calls,
call+typed-final, refusal/incomplete output and contradictory finish reasons.
Errors expose a fresh normalized exception with fixed text and suppressed raw
cause, including when a provider raises an existing gateway exception. Usage is
preserved on invalid output when available; tool text is not double-encoded.

### Compatibility gate discovered

The current public OrchestratorFinalResponse schema includes open dictionaries
and discriminated unions that this strict subset does not support. A dedicated
test verifies unavailability before provider construction. Task 10 must supply
a bounded model-facing final schema plus verified projection to existing public
blocks, and an actual-adapter compatibility test, before candidate admission.
This adapter passing with bounded test schemas does NOT prove the current
public block union or configured hosted model works. No change to public API or
silent conversion of oneOf to anyOf was made here.

### Official sources and inspected versions

Checked 2026-11-16:

- [OpenAI Function calling](https://developers.openai.com/api/docs/guides/function-calling): native calls/results and IDs; strict declarations and disabling parallel calls. Adopt these controls, retain WorkCrew policy guards separately.
- [OpenAI Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs): supported subset, required fields, nullable optional fields and closed objects. Reject unsupported contracts instead of dropping their constraints.
- Installed source: langchain-openai 1.4.1 `chat_models/base.py` bind_tools/response_format; langchain-core 1.5.3 function_calling.py strict conversion; OpenAI SDK 2.52.1 and Pydantic 2.13.4. Versions remain pinned by ai/uv.lock. No upgrade.

### Verification

- RED initially: missing openai_tools module. Additional negative tests observed
  raw text escaping via an already-normalized provider error, inconsistent
  finish metadata and double-encoded text observation before fixes.
- Targeted new native + existing gateway/media tests: 61 PASS (28 native).
- Includes two tests traversing the real installed LangChain/OpenAI SDK stack
  via httpx.MockTransport: captured stateless HTTP payload, strict schemas,
  output cap and call/result IDs; normalized provider tool call or typed final.
  These are offline transport tests, not hosted model quality measurements.
- make ai: lint/format PASS, pyright 0 errors, 604 passed / 3 live-provider tests
  deselected. Core eval still exits nonzero: progressive planning 14 PASS and
  6 DB cases NOT_RUN, other six core gates PASS; full AI gate not green.
- make -C ai eval-retrieve separately: PASS, required_total 28, safety 0,
  hosted NOT_RUN. No database reset or hosted smoke (reserved for Task 15).
