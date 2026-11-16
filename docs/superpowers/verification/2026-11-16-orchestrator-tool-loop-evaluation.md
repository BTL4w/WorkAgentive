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
