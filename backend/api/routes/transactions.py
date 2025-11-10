from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from backend.api.routes.auth import get_current_user
from backend.utils.db_writer import add_transaction_record
from backend.agents.transaction_classifier import TransactionClassifierAgent
from backend.utils.csv_parser import parse_csv_bytes
from fastapi import Request

from backend.agents.sql_nl_agent import text_to_sql
from backend.agents.analytics_agent import analytics_agent
transactions_router = APIRouter()
transaction_agent= TransactionClassifierAgent()

# Endpoint to upload transactions from a CSV file
@transactions_router.post("/upload-transactions")
async def uploaded_transactions(file: UploadFile = File(...), user_id=Depends(get_current_user)):
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

    '''
    # --- CACHE INVALIDATION ---
    cache_key = f"dashboard_{user_id}"
    if analytics_agent.memory.get(cache_key):
        analytics_agent.memory.delete(cache_key)
    '''

    return {"message": f"{len(results)} transactions stored successfully."}


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