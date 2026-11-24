"""Typed human-gated automation preparation contracts."""

from typing import Literal

from pydantic import Field

from work_management_ai.agents.orchestrator.contracts import ScheduleIntent
from work_management_ai.runtime.contracts import DailySummaryResponseBlock

AutomationToolInput = ScheduleIntent
AutomationToolOutput = DailySummaryResponseBlock


class AutomationToolInputV2(ScheduleIntent):
    """Schema sent to the candidate native tool gateway."""

    project_reference: str = Field(
        min_length=1, max_length=200, description="Project name or verified reference."
    )
    operation: Literal["CONFIGURE", "PAUSE", "RESUME"] = Field(
        description="Requested schedule preview operation."
    )
    timezone: str | None = Field(default=None, description="Requested IANA timezone, if supplied.")
    cutoff: str | None = Field(default=None, description="Requested local reporting cutoff.")
    weekdays: tuple[int, ...] | None = Field(
        default=None, description="Requested weekdays using ISO weekday numbering."
    )
    recipient_references: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Requested recipient names or references.",
    )
    send_when_complete: bool | None = Field(
        default=None, description="Whether to send once reporting is complete."
    )
    partial_at_cutoff: bool | None = Field(
        default=None, description="Whether to send an incomplete digest at the cutoff."
    )
    narrative_mode: Literal["NONE", "DRAFT_FOR_MANAGER"] | None = Field(
        default=None, description="Requested narrative mode."
    )
    narrative_locale: Literal["vi", "en"] | None = Field(
        default=None, description="Requested narrative language."
    )
