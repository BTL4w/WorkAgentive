"""Trusted user clarification receipts supplied by the application runtime."""

from typing import Protocol
from uuid import UUID

from pydantic import Field

from work_management_ai.runtime.tool_loop_contracts import RunScope, VersionPin
from work_management_ai.tools.context.contracts import Contract


class ResolvedContextBinding(Contract):
    source_scope: RunScope
    user_turn_id: UUID
    source_invocation_id: UUID
    query_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_tool_pin: VersionPin
    selected_ref_id: UUID
    resource_id: UUID
    resource_version: int = Field(ge=1, strict=True)
    resource_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    actor_revision: str = Field(min_length=1, max_length=128)
    policy_revision: str = Field(min_length=1, max_length=128)


class UserClarificationPort(Protocol):
    async def accepted(
        self, *, source_scope: RunScope, user_turn_id: UUID, source_invocation_id: UUID
    ) -> ResolvedContextBinding | None:
        """Read a server-verified choice from an accepted user turn."""
        ...
