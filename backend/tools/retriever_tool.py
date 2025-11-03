
from dotenv import load_dotenv
import json
from backend.utils.db_connector import engine
from sqlalchemy import text
import pandas as pd
load_dotenv()

def fetch_user_transactions(user_id: str) -> pd.DataFrame:
    """Fetch all transactions for a user from the database."""
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
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')

    return df
