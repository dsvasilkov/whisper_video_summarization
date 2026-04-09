"""drop legacy result_transcription text

Revision ID: c3d2e1f0a9b8
Revises: b9e7c6d5a4f3
Create Date: 2026-04-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d2e1f0a9b8"
down_revision: Union[str, None] = "b9e7c6d5a4f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("inference_tasks", "result_transcription")


def downgrade() -> None:
    op.add_column(
        "inference_tasks",
        sa.Column("result_transcription", sa.Text(), nullable=True),
    )
