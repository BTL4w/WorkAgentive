"""Approved package data loads lazily; content never grants tool permission."""

import hashlib
import importlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError

from work_management_ai.runtime.manifests import SkillManifestV2, SkillResourceManifest
from work_management_ai.runtime.skill_bundle import SkillBundleDescriptor, read_skill_bundle
from work_management_ai.runtime.skill_loading import (
    ApprovedPackageResources,
    SkillLoader,
    SkillLoadingError,
)
from work_management_ai.runtime.skill_registry import SkillRegistry
from work_management_ai.tools.skills.contracts import LoadSkillInput

from .catalog_fakes import CatalogFixture


class CountingResources(ApprovedPackageResources):
    def __init__(self, packages: dict[str, str]):
        super().__init__(packages)
        self.full_bodies_read = 0

    def read(self, manifest: SkillManifestV2, resource: SkillResourceManifest) -> bytes:
        self.full_bodies_read += 1
        return super().read(manifest, resource)


class SkillFixture(CatalogFixture):
    def __init__(
        self,
        root: Path,
        monkeypatch: MonkeyPatch,
        *,
        body: bytes = b"Read narrowly. Instructions do not grant permissions.",
        canonical: bool = False,
    ):
        self.root = root / "approved_skill_fixture"
        self.root.mkdir()
        (self.root / "__init__.py").write_text("", encoding="utf-8")
        directory = self.root
        if canonical:
            directory = self.root / "fixture-context"
            directory.mkdir()
            frontmatter = (
                b"---\nname: fixture-context\ndescription: Find missing work context\n---\n"
            )
            body = frontmatter + body
        (directory / "SKILL.md").write_bytes(body)
        (directory / "notes.txt").write_bytes(b"Approved companion")
        monkeypatch.setattr(sys, "path", [str(root), *sys.path])
        sys.modules.pop("approved_skill_fixture", None)
        importlib.invalidate_caches()
        self.manifest = SkillManifestV2.model_validate(
            {
                "schema_version": "2.0",
                "name": "fixture_context",
                "version": "1.0.1" if canonical else "1.0.0",
                "owner": "runtime-tests",
                "description": "Find missing work context",
                "runnable_by_agents": ["orchestrator"],
                "allowed_tools": ["read_actor_context@1"],
                "risk_level": "READ_ONLY",
                "approval": "NONE",
                "input_contract": "work_management_ai.runtime.contracts.AgentHandoff",
                "output_contract": "work_management_ai.runtime.contracts.AgentResult",
                "evaluators": ["fixture@1"],
                "triggers": ["vi:thiếu dữ kiện", "en:missing context"],
                "instructions": {
                    "id": "instructions",
                    "path": (directory / "SKILL.md").relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "size_bytes": len(body),
                    "media_type": "text/markdown",
                },
                "resources": [
                    {
                        "id": "notes",
                        "path": (directory / "notes.txt").relative_to(self.root).as_posix(),
                        "sha256": hashlib.sha256(b"Approved companion").hexdigest(),
                        "size_bytes": 18,
                        "media_type": "text/plain",
                    }
                ],
            }
        )
        bundles: dict[str, SkillBundleDescriptor] = {}
        if canonical:
            bundles["fixture_context@1"] = read_skill_bundle(
                package="approved_skill_fixture",
                instructions=self.manifest.instructions,
                resources=self.manifest.resources,
            )
        super().__init__(skill_registry=SkillRegistry([self.manifest], bundles=bundles))
        self.resources = CountingResources({"fixture_context@1": "approved_skill_fixture"})
        self.loader = SkillLoader(
            skills=self.skills,
            resources=self.resources,
            catalog=self.catalog,
            model_ref="configured-model",
        )


def test_catalog_metadata_does_not_read_full_body(tmp_path: Path, monkeypatch: MonkeyPatch):
    f = SkillFixture(tmp_path, monkeypatch)
    entries = f.loader.catalog(f.scope)
    assert len(entries) == 1 and entries[0].reference == "fixture_context@1"
    assert "content" not in type(entries[0]).model_fields
    f.catalog.prepare(f.scope, model_ref="configured-model")  # self-contained turn needs no load
    assert f.resources.full_bodies_read == 0


def test_canonical_catalog_and_selected_body(tmp_path: Path, monkeypatch: MonkeyPatch):
    f = SkillFixture(tmp_path, monkeypatch, canonical=True)
    (entry,) = f.loader.catalog(f.scope)
    assert entry.name == "fixture-context"
    assert entry.description == "Find missing work context"
    assert entry.instructions_sha256 == f.manifest.instructions.sha256
    assert f.resources.full_bodies_read == 0

    selector = LoadSkillInput(skill_ref="fixture_context@1")
    output = f.loader.load(f.scope, selector)
    assert output.content == "Read narrowly. Instructions do not grant permissions."
    assert output.body_sha256 != entry.instructions_sha256
    assert f.resources.full_bodies_read == 1
    f.loader.revalidate_loaded(f.scope, selector, output)
    with pytest.raises(SkillLoadingError, match="SKILL_VERSION_CONFLICT"):
        f.loader.revalidate_loaded(
            f.scope, selector, output.model_copy(update={"content": "Unapproved instructions"})
        )


def test_canonical_bundle_rejects_pre_migration_pin(tmp_path: Path, monkeypatch: MonkeyPatch):
    f = SkillFixture(tmp_path, monkeypatch, canonical=True)
    selector = LoadSkillInput(skill_ref="fixture_context@1")
    current = f.loader.load(f.scope, selector)
    stale = current.pin.model_copy(update={"version": "1.0.0"})
    resumed = SkillLoader(
        skills=f.skills,
        resources=f.resources,
        catalog=f.catalog,
        model_ref="configured-model",
        known_loaded=dict([(f.scope, (stale,))]),
    )

    with pytest.raises(SkillLoadingError, match="SKILL_VERSION_CONFLICT") as error:
        resumed.load(f.scope, selector)

    assert error.value.stop_segment


def test_nested_bundle_cannot_load_without_registered_descriptor(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch, canonical=True)
    loader = SkillLoader(
        skills=SkillRegistry([f.manifest]),
        resources=f.resources,
        catalog=f.catalog,
        model_ref="configured-model",
    )

    with pytest.raises(SkillLoadingError, match="SKILL_BUNDLE_DESCRIPTOR_REQUIRED"):
        loader.catalog(f.scope)
    with pytest.raises(SkillLoadingError, match="SKILL_BUNDLE_DESCRIPTOR_REQUIRED"):
        loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


def test_full_body_and_index_are_pinned_without_expanding_tools(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch)
    f.revoke("read_actor_context@1")
    value = f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    assert value.version == "1.0.0" and value.content.startswith("Read narrowly")
    assert value.body_sha256 == hashlib.sha256(value.content.encode()).hexdigest()
    assert value.effective_tools == ()
    assert [r.id for r in value.resources] == ["notes"]
    assert f.resources.full_bodies_read == 1
    notes = f.loader.load(
        f.scope, LoadSkillInput(skill_ref="fixture_context@1", resource_id="notes")
    )
    assert notes.content == "Approved companion"
    assert f.resources.full_bodies_read == 2


def test_companion_requires_loaded_instructions_and_unknown_resources_fail(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch)
    with pytest.raises(SkillLoadingError, match="SKILL_INSTRUCTIONS_NOT_LOADED"):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1", resource_id="notes"))
    with pytest.raises(SkillLoadingError):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="missing@1"))
    f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    with pytest.raises(SkillLoadingError, match="SKILL_RESOURCE_NOT_DECLARED"):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1", resource_id="other"))


def test_revocation_wins_loaded_pin_and_actor_cache(tmp_path: Path, monkeypatch: MonkeyPatch):
    from uuid import uuid4

    f = SkillFixture(tmp_path, monkeypatch)
    f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    f.access.value = replace(
        f.access.value, allowed_skills=frozenset(), permission_revision="revoked"
    )
    with pytest.raises(SkillLoadingError):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    assert f.loader.catalog(f.scope) == ()
    other = f.scope.model_copy(update={"membership_id": uuid4()})
    f.access.value = replace(
        f.access.value, scope=other, allowed_skills=frozenset({"fixture_context@1"})
    )
    with pytest.raises(SkillLoadingError, match="SKILL_INSTRUCTIONS_NOT_LOADED"):
        f.loader.load(other, LoadSkillInput(skill_ref="fixture_context@1", resource_id="notes"))


@pytest.mark.parametrize(
    "selector",
    ["../SKILL.md", "/tmp/data", "https://example.test/x", "a/../../x", "a\\b", "%2e%2e", ".."],
)
def test_arbitrary_resource_selector_rejected(selector: str):
    with pytest.raises(ValidationError):
        LoadSkillInput(skill_ref="fixture_context@1", resource_id=selector)


@pytest.mark.parametrize("body", [b"a" * 16384, "Việt".encode() * 2048])
def test_complete_utf8_body_up_to_16k(tmp_path: Path, monkeypatch: MonkeyPatch, body: bytes):
    f = SkillFixture(tmp_path, monkeypatch, body=body)
    assert (
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1")).content.encode()
        == body
    )


@pytest.mark.parametrize(
    "body,code",
    [
        (b"a" * 16385, "SKILL_RESOURCE_TOO_LARGE"),
        (b"wrong digest", "SKILL_RESOURCE_MISMATCH"),
        (b"\xff", "SKILL_RESOURCE_MISMATCH"),
    ],
)
def test_modified_or_oversized_body_never_returns_partial_success(
    tmp_path: Path, monkeypatch: MonkeyPatch, body: bytes, code: str
):
    f = SkillFixture(tmp_path, monkeypatch)
    (f.root / "SKILL.md").write_bytes(body)
    with pytest.raises(SkillLoadingError, match=code):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


def test_valid_digest_invalid_utf8_and_symlink_are_denied(tmp_path: Path, monkeypatch: MonkeyPatch):
    f = SkillFixture(tmp_path, monkeypatch, body=b"\xff")
    with pytest.raises(SkillLoadingError, match="SKILL_RESOURCE_ENCODING"):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    outside = tmp_path / "outside.txt"
    outside.write_text("private outside", encoding="utf-8")
    (f.root / "SKILL.md").unlink()
    (f.root / "SKILL.md").symlink_to(outside)
    with pytest.raises(SkillLoadingError):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


def test_catalog_unavailable_resource_fails_preparation_not_empty_catalog(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch)
    (f.root / "notes.txt").unlink()
    with pytest.raises(SkillLoadingError, match="SKILL_RESOURCE_UNAVAILABLE"):
        f.loader.catalog(f.scope)
    assert f.resources.full_bodies_read == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [12, 16384])
async def test_actual_loader_uses_pipeline_and_records_history_pin(
    tmp_path: Path, monkeypatch: MonkeyPatch, length: int
):
    from datetime import UTC, datetime, timedelta
    from uuid import UUID, uuid4

    from pydantic import BaseModel

    from work_management_ai.model_gateway.tool_contracts import ToolCall
    from work_management_ai.runtime.contracts import ContextReference
    from work_management_ai.runtime.run_budget import CandidateBudget, candidate_budget
    from work_management_ai.runtime.session_history import HistoryLimits, SessionHistory
    from work_management_ai.runtime.skill_loading import SkillLoaderBackend
    from work_management_ai.runtime.tool_loop_contracts import (
        AdmittedTool,
        BackendToolResult,
        RunLease,
        RunScope,
        ToolInvocationContext,
        ToolLoopCheckpoint,
        VersionPin,
    )
    from work_management_ai.runtime.tool_pipeline import PreparedTool, ToolPipeline

    from .test_tool_loop_budget import Budget
    from .test_tool_loop_history import Access
    from .test_tool_pipeline import Log

    f = SkillFixture(tmp_path, monkeypatch, body=b"a" * length)
    log = Log()

    class Policy:
        async def pre_execute(
            self, scope: RunScope, tool: AdmittedTool, arguments: BaseModel
        ) -> PreparedTool:
            return PreparedTool(arguments=arguments, policy_version="test-policy")

        async def guard(
            self,
            context: ToolInvocationContext,
            tool: AdmittedTool,
            evidence: tuple[ContextReference, ...],
        ) -> None:
            selected = f.catalog.prepare(context.scope, model_ref="configured-model")
            try:
                selected.resolve("load_skill")
            finally:
                selected.close()

        async def post_execute(
            self, context: ToolInvocationContext, result: BackendToolResult
        ) -> None:
            pass

        async def rejected(
            self, scope: RunScope, invocation_id: UUID, tool_reference: str | None, safe_code: str
        ) -> None:
            pass

    lease = RunLease(
        scope=f.scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="test",
        fencing_token=1,
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    budget = Budget(f.scope, lease)
    backend = SkillLoaderBackend(f.loader)
    pipeline = ToolPipeline(store=log, policy=Policy(), backends={"load_skill@1": backend})
    admitted = f.catalog.prepare(f.scope, model_ref="configured-model")
    call = ToolCall(
        call_id="load-1",
        name="load_skill",
        arguments={"skill_ref": "fixture_context@1", "resource_id": None},
    )
    with candidate_budget(
        CandidateBudget(scope=f.scope, lease=lease, store=budget, interval_id=uuid4())
    ):
        first = await pipeline.execute(f.scope, call, admitted=admitted, lease=lease)
        assert first.kind == "COMPLETED"
        payload = await log.get_payload(f.scope, first.payload_ref)
        assert isinstance(payload, dict) and payload["loaded_skill"] is not None
        assert await pipeline.execute(f.scope, call, admitted=admitted, lease=lease) == first
    assert f.resources.full_bodies_read == 1
    checkpoint = ToolLoopCheckpoint(
        scope=f.scope,
        driver_version="2.0.0",
        prompt_version="test",
        catalog_fingerprint=admitted.fingerprint,
        tool_versions=(
            VersionPin(
                reference="load_skill@1",
                version=admitted.resolve("load_skill").version,
                fingerprint=admitted.resolve("load_skill").fingerprint,
            ),
        ),
        skill_versions=(VersionPin.model_validate(payload["loaded_skill"]),),
    )
    history = await SessionHistory(log).project(
        f.scope,
        tuple(log.events),
        checkpoint=checkpoint,
        access=Access(),
        limits=HistoryLimits(),
        skill_access=f.loader,
    )
    assert history.status == "READY" and len(history.loaded_skills) == 1
    assert history.loaded_skills[0].reference == "fixture_context@1"
    original_access = f.access.value
    original_revalidate = f.loader.revalidate_skills

    async def revoke_during_projection(scope: RunScope, pins: tuple[VersionPin, ...]):
        verdict = await original_revalidate(scope, pins)
        f.access.value = replace(
            f.access.value, allowed_skills=frozenset(), permission_revision="mid-read"
        )
        return verdict

    f.loader.revalidate_skills = revoke_during_projection
    during = await SessionHistory(log).project(
        f.scope,
        tuple(log.events),
        checkpoint=checkpoint,
        access=Access(),
        limits=HistoryLimits(),
        skill_access=f.loader,
    )
    assert during.status == "CONTEXT_REVOKED" and during.items == ()
    f.loader.revalidate_skills = original_revalidate
    f.access.value = original_access
    f.access.value = replace(
        f.access.value, allowed_skills=frozenset(), permission_revision="revoked"
    )
    fresh_catalog = f.catalog.prepare(f.scope, model_ref="configured-model")
    with candidate_budget(
        CandidateBudget(scope=f.scope, lease=lease, store=budget, interval_id=uuid4())
    ):
        denied = await pipeline.execute(f.scope, call, admitted=fresh_catalog, lease=lease)
    assert denied.kind == "REJECTED" and denied.stop_segment
    assert f.resources.full_bodies_read == 1
    revoked_history = await SessionHistory(log).project(
        f.scope,
        tuple(log.events),
        checkpoint=checkpoint,
        access=Access(),
        limits=HistoryLimits(),
        skill_access=f.loader,
    )
    assert revoked_history.status == "CONTEXT_REVOKED" and revoked_history.items == ()
    missing_guard = await SessionHistory(log).project(
        f.scope, tuple(log.events), checkpoint=checkpoint, access=Access(), limits=HistoryLimits()
    )
    assert missing_guard.status == "CONTEXT_UNAVAILABLE" and missing_guard.items == ()


def test_resume_pin_rehydrates_only_exact_approved_version(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch)
    loaded = f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
    resumed = SkillLoader(
        skills=f.skills,
        resources=f.resources,
        catalog=f.catalog,
        model_ref="configured-model",
        known_loaded=dict([(f.scope, (loaded.pin,))]),
    )
    assert (
        resumed.load(
            f.scope, LoadSkillInput(skill_ref="fixture_context@1", resource_id="notes")
        ).content
        == "Approved companion"
    )
    stale = SkillLoader(
        skills=f.skills,
        resources=f.resources,
        catalog=f.catalog,
        model_ref="configured-model",
        known_loaded=dict([(f.scope, (loaded.pin.model_copy(update={"version": "2.0.0"}),))]),
    )
    with pytest.raises(SkillLoadingError, match="SKILL_VERSION_CONFLICT"):
        stale.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


def test_valid_digest_with_control_characters_has_safe_encoding_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch, body=b"instructions\x00private")
    with pytest.raises(SkillLoadingError, match="SKILL_RESOURCE_ENCODING"):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


def test_revocation_during_body_read_never_returns_content(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    from work_management_ai.runtime.manifests import SkillResourceManifest

    f = SkillFixture(tmp_path, monkeypatch)
    original = f.resources.read

    def revoked(manifest: SkillManifestV2, resource: SkillResourceManifest) -> bytes:
        body = original(manifest, resource)
        f.access.value = replace(
            f.access.value, allowed_skills=frozenset(), permission_revision="p2"
        )
        return body

    f.resources.read = revoked
    with pytest.raises(SkillLoadingError):
        f.loader.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))


@pytest.mark.asyncio
async def test_stale_catalog_call_stops_segment_before_skill_backend(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from work_management_ai.model_gateway.tool_contracts import ToolCall
    from work_management_ai.runtime.run_budget import CandidateBudget, candidate_budget
    from work_management_ai.runtime.skill_loading import SkillLoaderBackend
    from work_management_ai.runtime.tool_loop_contracts import RunLease
    from work_management_ai.runtime.tool_pipeline import ToolPipeline

    from .test_tool_loop_budget import Budget
    from .test_tool_pipeline import Log, Policy

    f = SkillFixture(tmp_path, monkeypatch)
    lease = RunLease(
        scope=f.scope,
        lease_id=uuid4(),
        job_id=uuid4(),
        worker_id="t",
        fencing_token=1,
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    budget = Budget(f.scope, lease)
    admitted = f.catalog.prepare(f.scope, model_ref="configured-model")
    pipeline = ToolPipeline(
        store=Log(), policy=Policy(), backends={"load_skill@1": SkillLoaderBackend(f.loader)}
    )
    f.revoke("load_skill@1")
    with candidate_budget(
        CandidateBudget(scope=f.scope, lease=lease, store=budget, interval_id=uuid4())
    ):
        result = await pipeline.execute(
            f.scope,
            ToolCall(
                call_id="stale", name="load_skill", arguments={"skill_ref": "fixture_context@1"}
            ),
            admitted=admitted,
            lease=lease,
        )
    assert result.kind == "REJECTED" and result.stop_segment
    assert f.resources.full_bodies_read == 0


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/etc/passwd",
        "C:\\outside.txt",
        "scripts/run.py",
        "x.sh",
        "https://example.test/a.md",
    ],
)
def test_approved_manifest_itself_rejects_arbitrary_resource_paths(path: str):
    from work_management_ai.runtime.manifests import SkillResourceManifest

    with pytest.raises(ValidationError):
        SkillResourceManifest(
            id="instructions", path=path, sha256="a" * 64, size_bytes=1, media_type="text/plain"
        )


def test_manifest_index_duplicate_ids_and_oversized_declaration_fail(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    f = SkillFixture(tmp_path, monkeypatch)
    data = f.manifest.model_dump(mode="json")
    data["resources"][0]["id"] = "instructions"
    with pytest.raises(ValidationError):
        SkillManifestV2.model_validate(data)
    data = f.manifest.model_dump(mode="json")
    data["instructions"]["size_bytes"] = 16385
    with pytest.raises(ValidationError):
        SkillManifestV2.model_validate(data)


@pytest.mark.parametrize(
    "contract",
    [
        "work_management_ai.runtime.contracts.RiskLevel",
        "work_management_ai.runtime.manifests.canonical_manifest_fingerprint",
    ],
)
def test_v2_skill_contract_must_be_typed_model(
    tmp_path: Path, monkeypatch: MonkeyPatch, contract: str
):
    from work_management_ai.runtime.skill_registry import SkillRegistryError

    f = SkillFixture(tmp_path, monkeypatch)
    data = f.manifest.model_dump(mode="json")
    data["input_contract"] = contract
    with pytest.raises(SkillRegistryError, match="SKILL_CONTRACT_INVALID"):
        SkillRegistry([SkillManifestV2.model_validate(data)])


def test_catalog_and_first_load_pin_approved_skill_dependency(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    from work_management_ai.runtime.agent_registry import AgentRegistry
    from work_management_ai.runtime.tool_catalog import ToolCatalog

    f = SkillFixture(tmp_path, monkeypatch)
    original = f.catalog.prepare(f.scope, model_ref="configured-model").fingerprint
    replacement = SkillRegistry([f.manifest.model_copy(update={"version": "1.1.0"})])
    registry = AgentRegistry(
        skill_registry=replacement,
        tool_registry=f.tools,
        evaluator_ids=frozenset({"orchestrator_plan@1", "fixture@1"}),
    )
    agent = registry.register_tool_loop(f.agent.manifest)
    changed = ToolCatalog(agent=agent, tools=f.tools, access=f.access, readiness=f.readiness)
    assert changed.prepare(f.scope, model_ref="configured-model").fingerprint != original
    mismatched = SkillLoader(
        skills=replacement, resources=f.resources, catalog=f.catalog, model_ref="configured-model"
    )
    with pytest.raises(SkillLoadingError, match="SKILL_VERSION_CONFLICT"):
        mismatched.load(f.scope, LoadSkillInput(skill_ref="fixture_context@1"))
