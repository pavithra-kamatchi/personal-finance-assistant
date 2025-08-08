from backend.api.models.schemas import TransactionDetails

from backend.utils.db_connector import engine

def add_transaction_record(transaction: TransactionDetails):
    """Update transaction record in the database."""

    # Connect to the database
    conn = engine.connect()
    # Create a cursor object
    cur = conn.cursor()

    # Insert the transaction record into the database
    insert_query = """
        INSERT INTO transactions (
            user_id,
            transaction_date,
            description,
            transaction_amount,
            category,
            merchant,
            account_name,
            type
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(insert_query, (
        transaction.user_id,
        transaction.transaction_date,
        transaction.description,
        transaction.transaction_amount,
        transaction.category,
        transaction.merchant,
        transaction.account_name,
        transaction.type
    ))
    conn.commit()
    cur.close()
    conn.close()
    