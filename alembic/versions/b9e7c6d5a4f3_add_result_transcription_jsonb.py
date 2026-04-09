"""add result_transcription_json JSONB

Revision ID: b9e7c6d5a4f3
Revises: 8a1c2d3e4f56
Create Date: 2026-04-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b9e7c6d5a4f3"
down_revision: Union[str, None] = "8a1c2d3e4f56"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inference_tasks",
        sa.Column("result_transcription_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("inference_tasks", "result_transcription_json")
