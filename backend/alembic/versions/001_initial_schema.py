"""Initial schema creation with all core tables.

Revision ID: 001
Revises:
Create Date: 2026-04-11 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'])

    # Create colors table
    op.create_table(
        'colors',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('rgb', sa.String(6), nullable=True),
        sa.Column('is_transparent', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )

    # Create part_categories table
    op.create_table(
        'part_categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )

    # Create parts table
    op.create_table(
        'parts',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('part_num', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('part_category_id', sa.Integer(), nullable=False),
        sa.Column('material', sa.String(50), nullable=True),
        sa.Column('image_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['part_category_id'], ['part_categories.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_parts_part_num', 'parts', ['part_num'])

    # Create themes table
    op.create_table(
        'themes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('parent_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['parent_id'], ['themes.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create lego_sets table
    op.create_table(
        'lego_sets',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('set_num', sa.String(50), nullable=False, unique=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('theme_id', sa.Integer(), nullable=False),
        sa.Column('num_parts', sa.Integer(), nullable=False),
        sa.Column('image_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['theme_id'], ['themes.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_lego_sets_set_num', 'lego_sets', ['set_num'])

    # Create set_parts table (join table between sets and parts)
    op.create_table(
        'set_parts',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('set_id', sa.UUID(), nullable=False),
        sa.Column('part_id', sa.UUID(), nullable=False),
        sa.Column('color_id', sa.Integer(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('is_spare', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['set_id'], ['lego_sets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['part_id'], ['parts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['color_id'], ['colors.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('set_id', 'part_id', 'color_id', name='uq_set_part_color')
    )

    # Create inventory_items table
    op.create_table(
        'inventory_items',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('part_id', sa.UUID(), nullable=False),
        sa.Column('color_id', sa.Integer(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['part_id'], ['parts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['color_id'], ['colors.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'part_id', 'color_id', name='uq_inventory_user_part_color')
    )
    op.create_index('ix_inventory_user', 'inventory_items', ['user_id'])
    op.create_index('ix_inventory_part', 'inventory_items', ['part_id'])

    # Create scan_logs table
    op.create_table(
        'scan_logs',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('part_id', sa.UUID(), nullable=True),
        sa.Column('color_id', sa.Integer(), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('image_path', sa.String(500), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='success'),
        sa.Column('error_message', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['part_id'], ['parts.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['color_id'], ['colors.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_scan_logs_user', 'scan_logs', ['user_id'])
    op.create_index('ix_scan_logs_created', 'scan_logs', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('ix_scan_logs_created', table_name='scan_logs')
    op.drop_index('ix_scan_logs_user', table_name='scan_logs')
    op.drop_table('scan_logs')

    op.drop_index('ix_inventory_part', table_name='inventory_items')
    op.drop_index('ix_inventory_user', table_name='inventory_items')
    op.drop_table('inventory_items')

    op.drop_table('set_parts')

    op.drop_index('ix_lego_sets_set_num', table_name='lego_sets')
    op.drop_table('lego_sets')

    op.drop_table('themes')

    op.drop_index('ix_parts_part_num', table_name='parts')
    op.drop_table('parts')

    op.drop_table('part_categories')
    op.drop_table('colors')

    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
