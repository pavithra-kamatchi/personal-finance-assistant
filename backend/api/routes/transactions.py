from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from backend.api.routes.auth import get_current_user
from backend.utils.db_writer import add_transaction_record
from backend.agents.transaction_classifier import TransactionClassifierAgent
from backend.utils.csv_parser import parse_csv_bytes
from fastapi import Request
from backend.utils.cache_manager import update_user_cache, cache
from backend.agents.sql_nl_agent import text_to_sql
from backend.agents.analytics_agent import generate_dashboard_analytics
transactions_router = APIRouter()
transaction_agent= TransactionClassifierAgent()
import pandas as pd

# Endpoint to upload transactions from a CSV file
@transactions_router.post("/upload-transactions")
async def uploaded_transactions(background_tasks: BackgroundTasks, file: UploadFile = File(...), user_id=Depends(get_current_user)):
    allowed_types = ["text/csv", "application/vnd.ms-excel", "application/csv"]
    if file.content_type not in allowed_types and not file.filename.lower().endswith(".csv"):
        # Check if the file is a CSV based on content type and extension
        print(f"File content type: {file.content_type}, filename: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be a CSV")

    contents = await file.read()
    parsed_data = parse_csv_bytes(contents)
    results = transaction_agent.process_uploaded_transactions(parsed_data, user_id=user_id)
    if not results:
        raise HTTPException(status_code=400, detail="No valid transactions found in the file")
    
    for tx in results:
        add_transaction_record(tx)

    #Update the cache with new transactions
    new_df = pd.DataFrame(results)
    try:
        update_user_cache(str(user_id), new_df)
    except Exception as e:
        print(f"Cache update failed: {e}")

    # Invalidate dashboard cache
    dashboard_key = f"user_dashboard_{user_id}"
    cache.delete(dashboard_key)
    print(f"🧹 Invalidated dashboard cache for {user_id}")

    background_tasks.add_task(generate_dashboard_analytics, user_id)
    print("Background task to regenerate dashboard analytics for {} started.".format(user_id))

    return {"message": f"{len(results)} transactions stored successfully and cache updated."}


# Endpoint to get transactions based on natural language query
@transactions_router.get("/query")
def get_transactions(nl_query: str, user_id: str = Depends(get_current_user)):
    try:
        results = text_to_sql(
            nl_query,
            str(user_id),
            [],  
        )
        return {
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))