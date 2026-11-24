"""The server admits a pinned candidate only within pilot configuration."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.modules.assistant.api.schemas import PostAssistantMessageRequest
from app.modules.assistant.domain.session import SessionError
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.organization.domain.roles import MembershipRole


def test_admission_keeps_routes_and_is_scoped_to_pilot_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.modules.assistant.adapters.tool_loop_runtime import (
        DriverAdmission,
        build_candidate_artifacts,
    )

    pilot = uuid4()
    monkeypatch.delenv("APP_ORCHESTRATOR_DRIVER", raising=False)
    monkeypatch.delenv("APP_ORCHESTRATOR_TOOL_LOOP_ORGANIZATIONS", raising=False)
    defaults = Settings()
    settings = Settings(
        orchestrator_driver="tool_loop",
        orchestrator_tool_loop_organizations=(pilot,),
        ai_provider="mock",
        ai_model="main-model",
        ai_title_model="auxiliary-model",
    )
    actor = AuthenticatedActor(
        user_id=uuid4(),
        email="actor@example.test",
        display_name="Minh",
        membership_id=uuid4(),
        organization_id=pilot,
        organization_name="Pilot",
        role=MembershipRole.MANAGER,
    )
    assert defaults.orchestrator_driver == "legacy"
    assert defaults.orchestrator_tool_loop_organizations == ()
    artifacts = build_candidate_artifacts()
    admission = DriverAdmission(settings, artifacts)
    pin = admission.select(actor, verified_action=False, continuation_source=None)
    assert pin is not None
    assert pin.driver == "tool_loop"
    assert pin.agent_version == "2.0.0"
    assert len(pin.tool_versions) >= 5 and len(pin.skill_versions) == 1
    assert pin.tool_versions[0].fingerprint
    assert pin.model_ref == "main-model"
    assert settings.ai_title_model == "auxiliary-model"
    assert admission.select(actor, verified_action=True, continuation_source=None) is None
    assert admission.select(actor, verified_action=False, continuation_source="legacy") is None
    foreign = AuthenticatedActor(
        user_id=actor.user_id,
        email=actor.email,
        display_name=actor.display_name,
        membership_id=uuid4(),
        organization_id=uuid4(),
        organization_name="Other",
        role=MembershipRole.MANAGER,
    )
    assert admission.select(foreign, verified_action=False, continuation_source=None) is None
    rollback = DriverAdmission(
        settings.model_copy(update={"orchestrator_driver": "legacy"}), artifacts
    )
    assert rollback.select(actor, verified_action=False, continuation_source=None) is None
    assert rollback.resolve_saved(pin) == pin
    for pin_name, pins in (
        ("tool_versions", pin.tool_versions),
        ("skill_bundle_fingerprints", pin.skill_bundle_fingerprints),
        ("specialist_versions", pin.specialist_versions),
    ):
        with pytest.raises(SessionError, match="DRIVER_PIN_MISMATCH"):
            rollback.resolve_saved(pin.model_copy(update={pin_name: pins[1:]}))
        changed = pins[0].model_copy(update={"fingerprint": "changed-with-the-same-version"})
        with pytest.raises(SessionError, match="DRIVER_PIN_MISMATCH"):
            rollback.resolve_saved(pin.model_copy(update={pin_name: (changed, *pins[1:])}))


@pytest.mark.parametrize("field", ["driver_pin", "orchestrator_driver", "organization_id"])
def test_message_contract_rejects_client_driver_and_tenant_selection(field: str) -> None:
    with pytest.raises(ValidationError) as invalid:
        PostAssistantMessageRequest.model_validate(
            {"message": "Hello", "locale": "en", field: str(uuid4())}
        )
    assert invalid.value.errors()[0]["loc"] == (field,)
    assert invalid.value.errors()[0]["type"] == "extra_forbidden"
