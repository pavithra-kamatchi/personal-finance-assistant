from fastapi import APIRouter, Depends, HTTPException
from backend.api.routes.auth import get_current_user, supabase
from dotenv import load_dotenv
from backend.api.models.schemas import BudgetCategory, BudgetRequest

load_dotenv()

budget_router = APIRouter()

#endpoint to set or update budget
@budget_router.post("")
async def set_budget(budget_data: BudgetRequest, user_id: str = Depends(get_current_user)):
    try:
        # Delete existing budget for this user
        supabase.table("budgets").delete().eq("user_id", user_id).execute()

        # Insert new budget records
        budget_records = []
        for cat in budget_data.categories:
            budget_records.append({
                "user_id": user_id,
                "category": cat.category,
                "budget_limit": cat.limit,
                "total_budget": budget_data.total_budget
            })

        if budget_records:
            supabase.table("budgets").insert(budget_records).execute()

        return {
            "status": "success",
            "message": f"Budget set successfully with {len(budget_records)} categories"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#endpoint to get current budget
@budget_router.get("")
async def get_budget(user_id: str = Depends(get_current_user)):
    try:
        result = supabase.table("budgets").select("*").eq("user_id", user_id).execute()

        if not result.data:
            return {
                "status": "success",
                "data": {
                    "total_budget": 0,
                    "categories": []
                }
            }

        # Group by category and return
        categories = [
            {
                "category": row["category"],
                "limit": row["budget_limit"]
            }
            for row in result.data
        ]

        total_budget = result.data[0]["total_budget"] if result.data else 0

        return {
            "status": "success",
            "data": {
                "total_budget": total_budget,
                "categories": categories
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
