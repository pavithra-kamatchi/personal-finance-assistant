from backend.api.models.schemas import TransactionDetails

from backend.utils.db_connector import engine

from sqlalchemy import text

def add_transaction_record(transaction: TransactionDetails):
    """Adds a transaction record to the database."""
    insert_query = text("""
        INSERT INTO transactions (
            user_id, transaction_date, description, transaction_amount,
            category, merchant, account_name, type
        ) VALUES (
            :user_id, :transaction_date, :description, :transaction_amount,
            :category, :merchant, :account_name, :type
        )
    """)
    with engine.begin() as conn:
        conn.execute(insert_query, {
            "user_id": transaction.user_id,
            "transaction_date": transaction.transaction_date,
            "description": transaction.description,
            "transaction_amount": transaction.transaction_amount,
            "category": transaction.category,
            "merchant": transaction.merchant,
            "account_name": transaction.account_name,
            "type": transaction.type,
        })
