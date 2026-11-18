"""Live authorized Project/current-actor reads; no mutation or personalized store."""

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.identity.domain.auth import AuthenticatedActor
from app.modules.work.application.project_service import ProjectService
from app.modules.work.domain.projects import Project, ProjectNotFoundError


class ContextReadError(Exception):
    """Safe context boundary code, with no raw storage errors."""


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectContextSnapshot(Snapshot):
    id: UUID
    name: str
    description: str | None
    version: int
    fingerprint: str


class ProjectReadSnapshot(Snapshot):
    projects: tuple[ProjectContextSnapshot, ...] = ()
    resolution: str
    selected_id: UUID | None = None
    outcome: str = "OK"
    requested: tuple[str, ...]
    returned: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    cursor: str | None = None
    query_fingerprint: str
    observed_at: AwareDatetime


class ActorReadSnapshot(Snapshot):
    membership_id: UUID
    display_name: str | None = None
    organization_name: str | None = None
    fingerprint: str
    requested: tuple[str, ...]
    returned: tuple[str, ...]
    gaps: tuple[str, ...]
    observed_at: AwareDatetime


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def project_snapshot(project: Project) -> ProjectContextSnapshot:
    return ProjectContextSnapshot(
        id=project.id,
        name=project.name,
        description=project.description,
        version=project.version,
        fingerprint=fingerprint(
            [
                str(project.id),
                project.version,
                project.name,
                project.description,
                project.updated_at.isoformat(),
            ]
        ),
    )


class ContextReadService:
    def __init__(self, *, actors: CurrentActorService, projects: ProjectService, cursor_key: bytes):
        if len(cursor_key) < 32:
            raise ValueError("CURSOR_KEY_TOO_SHORT")
        self._actors, self._projects, self._key = actors, projects, cursor_key

    async def current_actor(self, *, actor: AuthenticatedActor) -> AuthenticatedActor:
        current = await self._actors.resolve(
            organization_id=actor.organization_id, membership_id=actor.membership_id
        )
        if current is None or current.user_id != actor.user_id:
            raise ContextReadError("CONTEXT_DENIED")
        return current

    @staticmethod
    def _sections(sections: tuple[str, ...]) -> tuple[str, ...]:
        if not 1 <= len(sections) <= 8 or any(not s or len(s) > 64 for s in sections):
            raise ContextReadError("INVALID_SECTIONS")
        return tuple(dict.fromkeys(sections))

    def _encode(self, payload: dict[str, object]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        sig = hmac.digest(self._key, raw, "sha256")
        return base64.urlsafe_b64encode(raw + sig).decode()

    def _decode(self, cursor: str, binding: str) -> tuple[int, str]:
        try:
            if len(cursor) > 2048:
                raise ValueError
            data = base64.b64decode(cursor, altchars=b"-_", validate=True)
            raw, sig = data[:-32], data[-32:]
            if not hmac.compare_digest(hmac.digest(self._key, raw, "sha256"), sig):
                raise ValueError
            payload = json.loads(raw)
            if (
                payload["binding"] != binding
                or type(payload["page"]) is not int
                or not 2 <= payload["page"] <= 10000
            ):
                raise ValueError
            return payload["page"], payload["boundary"]
        except (ValueError, KeyError, TypeError):
            raise ContextReadError("INVALID_CURSOR") from None

    async def search_projects(
        self, *, actor: AuthenticatedActor, query: str, cursor: str | None, page_size: int
    ) -> ProjectReadSnapshot:
        current = await self.current_actor(actor=actor)
        query = query.strip()
        if not 1 <= len(query) <= 160 or type(page_size) is not int or not 1 <= page_size <= 20:
            raise ContextReadError("INVALID_QUERY")
        query_fp = fingerprint([query, "created_at_desc_id_asc"])
        binding = fingerprint(
            [
                str(current.organization_id),
                str(current.membership_id),
                current.role.value,
                query_fp,
                page_size,
            ]
        )
        page = 1
        if cursor:
            page, boundary = self._decode(cursor, binding)
            previous = await self._projects.list_projects(
                actor=current, query=query, page=page - 1, page_size=page_size
            )
            if (
                fingerprint([project_snapshot(p).model_dump(mode="json") for p in previous.items])
                != boundary
            ):
                raise ContextReadError("CURSOR_STALE")
        result = await self._projects.list_projects(
            actor=current, query=query, page=page, page_size=page_size
        )
        more = page * page_size < result.total
        values = tuple(project_snapshot(p) for p in result.items)
        # Partial discovery never promotes a single visible page to unique binding.
        resolution = (
            "AMBIGUOUS"
            if more or page > 1 or len(values) > 1
            else "UNIQUE"
            if len(values) == 1
            else "NOT_FOUND_OR_NOT_ACCESSIBLE"
        )
        next_cursor = (
            self._encode(
                {
                    "binding": binding,
                    "page": page + 1,
                    "boundary": fingerprint([p.model_dump(mode="json") for p in values]),
                }
            )
            if more
            else None
        )
        return ProjectReadSnapshot(
            projects=values,
            resolution=resolution,
            selected_id=values[0].id if values and resolution == "UNIQUE" else None,
            outcome="PARTIAL" if more else "OK",
            requested=("discovery",),
            returned=("discovery",),
            cursor=next_cursor,
            query_fingerprint=query_fp,
            observed_at=datetime.now(UTC),
        )

    async def read_project(
        self, *, actor: AuthenticatedActor, reference: UUID, sections: tuple[str, ...]
    ) -> ProjectReadSnapshot:
        current = await self.current_actor(actor=actor)
        sections = self._sections(sections)
        gaps = tuple(s for s in sections if s != "overview")
        query_fp = fingerprint([str(reference), sections])
        observed = datetime.now(UTC)
        try:
            project = await self._projects.get_project(actor=current, project_id=reference)
        except ProjectNotFoundError:
            return ProjectReadSnapshot(
                resolution="NOT_FOUND_OR_NOT_ACCESSIBLE",
                outcome="NEEDS_INPUT",
                requested=sections,
                gaps=gaps,
                query_fingerprint=query_fp,
                observed_at=observed,
            )
        # No unsupported content is fetched from future stores or substituted with inventions.
        return ProjectReadSnapshot(
            projects=(project_snapshot(project),) if "overview" in sections else (),
            resolution="UNIQUE",
            selected_id=project.id if "overview" in sections else None,
            outcome="PARTIAL" if gaps else "OK",
            returned=("overview",) if "overview" in sections else (),
            requested=sections,
            gaps=gaps,
            query_fingerprint=query_fp,
            observed_at=observed,
        )

    async def read_actor(
        self, *, actor: AuthenticatedActor, sections: tuple[str, ...]
    ) -> ActorReadSnapshot:
        current = await self.current_actor(actor=actor)
        sections = self._sections(sections)
        supported = "identity" in sections
        return ActorReadSnapshot(
            membership_id=current.membership_id,
            display_name=current.display_name if supported else None,
            organization_name=current.organization_name if supported else None,
            fingerprint=fingerprint(
                [
                    str(current.membership_id),
                    current.display_name,
                    current.organization_name,
                    current.role.value,
                ]
            ),
            requested=sections,
            returned=("identity",) if supported else (),
            gaps=tuple(s for s in sections if s != "identity"),
            observed_at=datetime.now(UTC),
        )
