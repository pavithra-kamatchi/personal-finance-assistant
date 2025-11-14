from diskcache import Cache
import pandas as pd

cache = Cache("./cache")
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

def update_user_cache(user_id: str, new_transactions: pd.DataFrame):
    """Merge new transactions into existing cache and refresh dashboard."""
    cache_key = f"user_transactions_{user_id}"
    dashboard_key = f"user_dashboard_{user_id}"
    
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        combined = pd.concat([cached_data, new_transactions], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id", "transaction_date", "transaction_amount"])
        combined = combined.sort_values("transaction_date", ascending=False)
        print(f"Cache updated for {user_id} with {len(new_transactions)} new transactions.")
    else:
        combined = new_transactions
        print(f"Cache initialized for {user_id}.")

    cache.set(cache_key, combined, expire=CACHE_EXPIRY_SECONDS)
    cache.delete(dashboard_key)
    print(f" Dashboard cache invalidated for {user_id}. Will regenerate on next request.")
