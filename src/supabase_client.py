"""
src/supabase_client.py -- connection only.
Creates and returns one reusable Supabase client.
Credentials come from .env (never hard-coded, never committed).
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()   # reads the .env file in your project root

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_client = None


def get_client() -> Client:
    """Lazily create and reuse a single Supabase client."""
    global _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit(
            "Missing SUPABASE_URL or SUPABASE_KEY. "
            "Put them in a .env file in your project root."
        )
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
