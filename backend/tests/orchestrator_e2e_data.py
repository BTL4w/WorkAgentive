"""Link a synthetic workload source without creating an approved Planning artifact."""

import asyncio
import sys
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import create_database_engine


def is_browser_database(database_url: str) -> bool:
    database = urlsplit(database_url)
    return (
        database.scheme == "postgresql+psycopg"
        and database.hostname in {"localhost", "127.0.0.1"}
        and database.path == "/work_management_orchestrator_e2e"
        and not database.query
        and not database.fragment
    )


async def seed_workload(organization_id: UUID, task_id: UUID, membership_id: UUID) -> None:
    settings = get_settings()
    if not is_browser_database(settings.database_url):
        raise RuntimeError("FIXTURE_REQUIRES_DEDICATED_LOCAL_ORCHESTRATOR_E2E")
    engine = create_database_engine(settings)
    try:
        async with engine.begin() as connection:
            linked = await connection.scalar(
                text("""
                    UPDATE tasks AS task
                    SET assignee_membership_id = member.id
                    FROM projects AS project, memberships AS member, users AS person
                    WHERE task.organization_id = :organization AND task.id = :task
                      AND task.assignee_membership_id IS NULL
                      AND task.project_id = project.id
                      AND project.organization_id = task.organization_id
                      AND project.name LIKE 'Browser launch %'
                      AND member.organization_id = task.organization_id
                      AND member.id = :member AND member.is_active
                      AND person.id = member.user_id
                      AND person.email_normalized = 'employee@example.test'
                    RETURNING task.id
                """),
                {"organization": organization_id, "task": task_id, "member": membership_id},
            )
            if linked != task_id:
                raise RuntimeError("SYNTHETIC_WORKLOAD_TASK_NOT_FOUND")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    organization, task, membership = (UUID(value) for value in sys.argv[1:])
    asyncio.run(seed_workload(organization, task, membership))
