from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from engine.core.db import get_engine
from engine.core.models import metadata

target_metadata = metadata


def run_migrations_offline() -> None:
    from engine import config

    context.configure(url=config.DATABASE_URL, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # An explicit sqlalchemy.url (e.g. a scratch DB in tests) overrides the shared app engine
    url = context.config.get_main_option("sqlalchemy.url")
    engine = create_engine(url, poolclass=NullPool) if url else get_engine()
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
