"""Journal envelopes cannot smuggle raw context or incomplete invocation state."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from work_management_ai.runtime.session_log import EventDraft


def test_tool_requested_requires_recoverable_invocation() -> None:
    with pytest.raises(ValidationError):
        EventDraft(event_id=uuid4(), kind="tool.requested")


@pytest.mark.parametrize("field", ["thought", "raw_prompt", "metadata", "approved"])
def test_event_header_has_no_arbitrary_authority_or_context_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        EventDraft.model_validate(
            {"event_id": uuid4(), "kind": "turn.started", field: {"secret": "not-stored"}}
        )
