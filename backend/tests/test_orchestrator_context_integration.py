"""Real PostgreSQL application visibility, live actor, API and RLS boundaries."""

import os
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.core.config import Settings
from app.core.database import create_database_engine, create_session_factory
from app.main import create_app
from app.modules.assistant.application.context_service import ContextReadError, ContextReadService
from app.modules.identity.adapters.auth_repository import SqlAlchemyAuthTransactionFactory
from app.modules.identity.api.dependencies import get_authenticated_actor
from app.modules.identity.application.current_actor_service import CurrentActorService
from app.modules.work.adapters.project_repository import SqlAlchemyProjectTransactionFactory
from app.modules.work.application.project_service import ProjectService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_INTEGRATION") != "1", reason="requires dedicated PostgreSQL"
    ),
]


@pytest.mark.asyncio
async def test_context_service_preserves_employee_visibility_current_actor_and_real_rls():
    settings = Settings(environment="test")
    engine = create_database_engine(settings)
    org, other, user, employee, member, emp_member, foreign_user, foreign_member = [
        uuid4() for _ in range(8)
    ]
    visible, hidden, foreign = [uuid4() for _ in range(3)]
    try:
        async with engine.begin() as c:
            for oid in (org, other):
                await c.execute(
                    text(
                        "INSERT INTO organizations(id,slug,name) VALUES(:id,:slug,'Context Tenant')"
                    ),
                    {"id": oid, "slug": f"context-{oid.hex}"},
                )
            for uid, mid, oid, role in [
                (user, member, org, "MANAGER"),
                (employee, emp_member, org, "EMPLOYEE"),
                (foreign_user, foreign_member, other, "MANAGER"),
            ]:
                await c.execute(
                    text(
                        "INSERT INTO users(id,email_normalized,email_display,"
                        "display_name,password_hash) "
                        "VALUES(:id,:email,:email,'Context Actor','hash')"
                    ),
                    {"id": uid, "email": f"{uid.hex}@example.test"},
                )
                await c.execute(
                    text(
                        "INSERT INTO memberships(id,organization_id,user_id,role) "
                        "VALUES(:id,:org,:user,CAST(:role AS membership_role))"
                    ),
                    {"id": mid, "org": oid, "user": uid, "role": role},
                )
            for pid, oid, mid in [
                (visible, org, member),
                (hidden, org, member),
                (foreign, other, foreign_member),
            ]:
                await c.execute(
                    text(
                        "INSERT INTO projects(id,organization_id,name,description,"
                        "created_by_membership_id,updated_by_membership_id) "
                        "VALUES(:id,:org,'Context Project','Untrusted instruction',:member,:member)"
                    ),
                    {"id": pid, "org": oid, "member": mid},
                )
            await c.execute(
                text(
                    "INSERT INTO tasks(id,organization_id,project_id,title,status,"
                    "assignee_membership_id,created_by_membership_id,updated_by_membership_id) "
                    "VALUES(:id,:org,:project,'Task','TO_DO',:employee,:manager,:manager)"
                ),
                {
                    "id": uuid4(),
                    "org": org,
                    "project": visible,
                    "employee": emp_member,
                    "manager": member,
                },
            )
        sessions = create_session_factory(engine)
        actors = CurrentActorService(SqlAlchemyAuthTransactionFactory(sessions))
        manager = await actors.resolve(organization_id=org, membership_id=member)
        emp = await actors.resolve(organization_id=org, membership_id=emp_member)
        assert manager and emp
        projects = ProjectService(SqlAlchemyProjectTransactionFactory(sessions))
        service = ContextReadService(actors=actors, projects=projects, cursor_key=b"i" * 32)
        page = await service.search_projects(actor=emp, query="Context", cursor=None, page_size=20)
        assert [p.id for p in page.projects] == [visible]
        for pid in (hidden, foreign, uuid4()):
            result = await service.read_project(actor=emp, reference=pid, sections=("overview",))
            assert result.resolution == "NOT_FOUND_OR_NOT_ACCESSIBLE" and not result.projects
        assert (
            await service.read_project(actor=manager, reference=foreign, sections=("overview",))
        ).resolution == "NOT_FOUND_OR_NOT_ACCESSIBLE"
        app = create_app(settings, project_service=projects)
        app.dependency_overrides[get_authenticated_actor] = lambda: emp
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get(f"/api/v1/projects/{foreign}")).status_code == 404
            assert (await client.get(f"/api/v1/projects/{hidden}")).status_code == 404
        async with engine.begin() as c:
            await c.execute(text("SET LOCAL ROLE app_runtime"))
            assert (
                await c.scalar(text("SELECT count(*) FROM projects WHERE id=:id"), {"id": foreign})
                == 0
            )
            await c.execute(
                text("SELECT set_config('app.organization_id',:org,true)"), {"org": str(org)}
            )
            assert (
                await c.scalar(text("SELECT count(*) FROM projects WHERE id=:id"), {"id": foreign})
                == 0
            )
            assert (
                await c.scalar(text("SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user"))
                is False
            )
        async with engine.begin() as c:
            await c.execute(
                text("UPDATE memberships SET is_active=false WHERE id=:id"), {"id": emp_member}
            )
        with pytest.raises(ContextReadError):
            await service.read_actor(actor=emp, sections=("identity",))
    finally:
        await engine.dispose()
