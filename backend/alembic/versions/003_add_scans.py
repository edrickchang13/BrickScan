"""Add scans table

Revision ID: 003
Revises: 002
Create Date: 2026-04-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scans",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_s3_key", sa.String(), nullable=True),
        sa.Column("prediction", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confirmed_part_num", sa.String(), nullable=True),
        sa.Column("set_num", sa.String(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scans_user_id", "scans", ["user_id"])
    op.create_index("ix_scans_prediction", "scans", ["prediction"])
    op.create_index("ix_scans_confirmed_part_num", "scans", ["confirmed_part_num"])
    op.create_index("ix_scans_set_num", "scans", ["set_num"])
    op.create_index("ix_scans_created_at", "scans", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_scans_created_at", table_name="scans")
    op.drop_index("ix_scans_set_num", table_name="scans")
    op.drop_index("ix_scans_confirmed_part_num", table_name="scans")
    op.drop_index("ix_scans_prediction", table_name="scans")
    op.drop_index("ix_scans_user_id", table_name="scans")
    op.drop_table("scans")
