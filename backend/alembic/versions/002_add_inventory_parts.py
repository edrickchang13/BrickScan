"""Add inventory_parts table

Revision ID: 002
Revises: 001
Create Date: 2026-04-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inventory_parts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("set_num", sa.String(), nullable=False),
        sa.Column("part_num", sa.String(), nullable=False),
        sa.Column("color_id", sa.Integer(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_spare", sa.Boolean(), nullable=False, server_default="false"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("set_num", "part_num", "color_id", "is_spare", name="uq_inventory_part"),
    )
    op.create_index("ix_inventory_parts_set_num", "inventory_parts", ["set_num"])
    op.create_index("ix_inventory_parts_part_num", "inventory_parts", ["part_num"])


def downgrade() -> None:
    op.drop_index("ix_inventory_parts_part_num", table_name="inventory_parts")
    op.drop_index("ix_inventory_parts_set_num", table_name="inventory_parts")
    op.drop_table("inventory_parts")
