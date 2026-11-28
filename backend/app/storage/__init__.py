from .base import Store, StoreError

__all__ = ["Store", "StoreError", "build_store"]


def build_store(settings) -> Store:
    """Create the configured store (imported lazily so SQLite users never need Supabase code paths)."""
    if settings.storage_backend == "supabase":
        from .supabase_store import SupabaseStore

        return SupabaseStore(settings.supabase_url, settings.supabase_service_role_key)
    from .sqlite_store import SQLiteStore

    return SQLiteStore(settings.sqlite_path)
