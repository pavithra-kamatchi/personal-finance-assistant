from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from backend.api.routes.auth import get_current_user, supabase
from backend.agents.analytics_agent import generate_dashboard_analytics
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

            # Fetch budget data from Supabase
            budget_data = None
            try:
                budget_result = supabase.table("budgets").select("category, budget_limit").eq("user_id", user_id).execute()
                if budget_result.data:
                    # Transform to dict format: {"Groceries": 500, "Dining": 300}
                    budget_data = {row["category"]: row["budget_limit"] for row in budget_result.data}
                    print(f"Fetched budget data for {len(budget_data)} categories")
            except Exception as e:
                print(f"Error fetching budget data: {e}")

            results = generate_dashboard_analytics(user_id, budget_data=budget_data)

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

