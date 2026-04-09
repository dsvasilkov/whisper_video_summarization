"""initial inference_tasks

Revision ID: 4f2a8c1b9e0d
Revises:
Create Date: 2026-03-29

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4f2a8c1b9e0d"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE taskstatus AS ENUM ('pending', 'processing', 'completed', 'failed');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE tasktype AS ENUM ('text', 'audio', 'audio_upload');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )

    taskstatus = postgresql.ENUM(
        "pending",
        "processing",
        "completed",
        "failed",
        name="taskstatus",
        create_type=False,
    )
    tasktype = postgresql.ENUM(
        "text",
        "audio",
        "audio_upload",
        name="tasktype",
        create_type=False,
    )

    op.create_table(
        "inference_tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", taskstatus, nullable=False),
        sa.Column("task_type", tasktype, nullable=False),
        sa.Column("input_path", sa.String(length=2048), nullable=True),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("result_transcription", sa.Text(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("inference_tasks")
    op.execute("DROP TYPE IF EXISTS tasktype")
    op.execute("DROP TYPE IF EXISTS taskstatus")
