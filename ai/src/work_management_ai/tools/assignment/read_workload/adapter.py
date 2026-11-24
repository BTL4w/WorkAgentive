"""Application-service-only workload Tool adapter."""

import json
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from work_management_ai.runtime.contracts import (
    ContextReference,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from work_management_ai.tools.assignment.read_workload.contracts import (
    ReadWorkloadApplicationPort,
    ReadWorkloadInput,
    ReadWorkloadOutput,
)


def workload_fingerprint(snapshot: ReadWorkloadOutput) -> str:
    rows = sorted(snapshot.workloads, key=lambda row: (row.project_week_id, row.membership_id))
    payload = snapshot.model_copy(update={"workloads": tuple(rows)}).model_dump(
        mode="json", exclude={"observed_at"}
    )
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ReadWorkloadToolAdapter:
    def __init__(self, *, application: ReadWorkloadApplicationPort) -> None:
        self._application = application

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        if request.tool_id != "assignment.read_workload" or request.tool_version != "1.0.0":
            return _rejected("TOOL_IDENTITY_MISMATCH")
        try:
            value = ReadWorkloadInput.model_validate(request.typed_input)
        except ValidationError:
            return _rejected("TOOL_INPUT_INVALID")
        output = await self._application.read_workload(actor=request.actor, value=value)
        if (
            output.organization_id != request.actor.organization_id
            or output.project_id != value.project_id
        ):
            return _rejected("TOOL_TENANT_MISMATCH")
        return ToolExecutionResult(
            status="SUCCEEDED",
            typed_output=output.model_dump(mode="json"),
            evidence=(
                ContextReference(
                    reference_id=uuid5(
                        NAMESPACE_URL,
                        f"project-workload:{output.project_id}:{output.observed_at.isoformat()}",
                    ),
                    organization_id=output.organization_id,
                    resource_type="WORKLOAD",
                    resource_id=output.project_id,
                    fingerprint=workload_fingerprint(output),
                    observed_at=output.observed_at,
                ),
            ),
        )


def _rejected(code: str) -> ToolExecutionResult:
    return ToolExecutionResult(status="REJECTED", typed_output={}, safe_error_code=code)
