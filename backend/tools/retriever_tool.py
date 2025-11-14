from dotenv import load_dotenv
import json
from diskcache import Cache
from backend.utils.db_connector import engine
from sqlalchemy import text
import pandas as pd
load_dotenv()
# Initialize cache
cache = Cache("./cache")

def fetch_user_transactions(user_id: str) -> pd.DataFrame:
    """Fetch and cache all transactions for a user."""
    cache_key = f"user_transactions_{user_id}"
    cached = cache.get(cache_key)

    if cached is not None:
        print(f" Loaded transactions for {user_id} from cache.")
        return cached

    print(f" Fetching transactions for {user_id} from database...")
    query = text("""
        SELECT
            id,
            user_id,
            transaction_date,
            description,
            transaction_amount,
            category,
            merchant,
            account_name,
            type
        FROM transactions
        WHERE user_id = :user_id
        ORDER BY transaction_date DESC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"user_id": user_id})

    if not df.empty:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")

    # Save to cache
    cache.set(cache_key, df, expire=3600)  # expires after 1 hour
    print(f"💾 Cached {len(df)} transactions for {user_id}.")
    return df