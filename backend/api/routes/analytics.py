from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from backend.api.routes.auth import get_current_user
from backend.agents.analytics_agent import get_dashboard_analytics
from backend.utils.cache_manager import cache
import time
import json

analytics_router = APIRouter()

@analytics_router.get("/dashboard")
def get_dashboard(user_id: str = Depends(get_current_user)):
    try:
        start = time.time()
        dashboard_key = f"user_dashboard_{user_id}"

        cached_dashboard = cache.get(dashboard_key)
        if cached_dashboard:
            print(f"Returning cached dashboard for {user_id}")
            cached = json.loads(cached_dashboard)
            cached["latency_seconds"] = round(time.time() - start, 2)
            return cached
        else:
            print(f"Regenerating dashboard for {user_id}")
            results = get_dashboard_analytics(user_id)

            cache.set(dashboard_key, json.dumps({
            "status": "success",
            "data": results
            }), expire=3600)


            latency = time.time() - start
            print(f"[Latency] Analytics computed in {latency:.2f}s")

            return {
                "status": "success",
                "data": results,
                "latency_seconds": latency
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

