# Orchestrator Task 0 — baseline evidence

Ngày đo: 2026-11-16. Base commit: `7dd4cb4eb8004adfbdfb1268bc20f94e8f41902a`.
Branch: `feat/orchestrator-task-0`; measurement trên working tree có Task 0.
Unrelated AGENTS/retrieval changes đã có từ trước, không thuộc commit Task 0.

## Phạm vi

- Source amendment, authored synthetic bilingual corpus và kiểm tra hiện trạng.
- Không thay runtime/provider/model config, activation hoặc phase completion.
- Legacy suites dùng deterministic mock adapters/fixtures, không credentials.
- Candidate corpus: 60 cases/30 scenario pairs; **chưa chạy candidate driver**.
  Vocabulary allowed_outcomes/predicates là evaluation contract, không API enum.
- Working tree Task 1 draft test đã được giữ ở ignored scratch, không ảnh hưởng
  baseline hoặc được tính thành deliverable Task 0.

## Commands và kết quả

| Command | Kết quả đã quan sát |
| --- | --- |
| `cd ai && uv run --locked pytest tests/test_orchestrator_tool_loop_cases.py -q -m 'not live_provider'` trước corpus | FAIL, FileNotFoundError vì chưa có JSONL |
| Cùng command sau corpus | PASS, 1 passed |
| `make ai` lần đầu | Dừng ở ruff E501 trong test mới; đã format, sau đó chạy lại full check |
| Full AI check: `make check` trong `ai/` | Ruff lint/format PASS; pyright 0 errors; pytest 541 passed, 3 deselected; core eval exit 1 như bảng dưới |
| `cd ai && uv run --locked python -m work_management_ai.evaluation.retrieve_web` | Exit 0, required_total=28, passed=true, safety_violations=0, hosted_status=NOT_RUN |

`make ai` root và `make check` trong ai đều gọi cùng isolated check chain.
Pyright có thông báo update version; không tự upgrade dependency.

## Core MVP gate

| Suite | Kết quả |
| --- | --- |
| phase2_multi_agent | PASS |
| phase3_assignment | PASS |
| phase4_daily_update | PASS |
| phase4_risk | PASS |
| phase4_daily_update_risk | PASS |
| phase5_reporting | PASS |
| progressive_planning | graph_passed=true; 14 PASS, 6 NOT_RUN; overall false |

Cases chưa chạy: `refresh-vi/en`, `stale-approval-vi/en`, `permissions-vi/en`.
`measure_postgres_boundaries()` chỉ chạy khi `RUN_POSTGRES_INTEGRATION=1`;
không có opt-in đó trong baseline này. Vì vậy **full make ai gate chưa đạt**.
Không sửa unrelated evaluator hoặc bỏ cases để báo pass. Không reset database.

Retrieve suite ghi `AUTHORED_SYNTHETIC_NOT_HUMAN_REVIEWED_HELD_OUT`; hosted
metrics/routing accuracy chưa đo. Mock success không chứng minh hosted quality.

## Kiểm tra tài liệu và governance

Amendment thay mandatory DAG bằng bounded model-next-action, optional working
plan, narrow application-service exploration và private history/recovery log.
Giữ tenant isolation, authorization/RLS, exact-version approval, manual flows,
hub-and-spoke, activation và mọi phase status hiện tại. Không bật personalization,
general script/tool search, integrations hoặc direct autonomous business writes.

Commit chỉ bao gồm Task 0 corpus/test, source documents, nhật ký và evidence.
Spec/implementation plan local vẫn không được tự đưa thêm vào commit.

## Giới hạn và điều kiện so sánh sau này

1. New corpus chỉ có schema/pairing test; Task 14 mới implement candidate runner.
2. Chưa có hosted evidence hoặc performance/cost benchmark candidate.
3. PostgreSQL boundary evaluation còn chưa đo trong lần baseline này.
4. Không đánh dấu Core MVP/new runtime complete từ record này.
