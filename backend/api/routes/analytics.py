from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from backend.api.routes.auth import get_current_user
from backend.agents.analytics_agent import get_dashboard_analytics
import time

analytics_router = APIRouter()

@analytics_router.get("/dashboard")
def get_dashboard(user_id: str = Depends(get_current_user)):
    try:
        start = time.time()
        results = get_dashboard_analytics(user_id)

        latency = time.time() - start
        print(f"[Latency] Analytics computed in {latency:.2f}s")

        return {
            "status": "success",
            "data": results,
            "latency_seconds": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

