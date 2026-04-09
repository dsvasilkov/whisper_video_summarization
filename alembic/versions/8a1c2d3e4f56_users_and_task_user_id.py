"""users table and inference_tasks.user_id

Revision ID: 8a1c2d3e4f56
Revises: 4f2a8c1b9e0d
Create Date: 2026-04-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "8a1c2d3e4f56"
down_revision: Union[str, None] = "4f2a8c1b9e0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("password_reset_token_hash", sa.String(length=64), nullable=True),
        sa.Column("password_reset_expires", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.add_column(
        "inference_tasks",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        op.f("ix_inference_tasks_user_id"),
        "inference_tasks",
        ["user_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_inference_tasks_user_id_users",
        "inference_tasks",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_inference_tasks_user_id_users", "inference_tasks", type_="foreignkey")
    op.drop_index(op.f("ix_inference_tasks_user_id"), table_name="inference_tasks")
    op.drop_column("inference_tasks", "user_id")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
