from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.tools.retriever_tool import fetch_user_transactions
import json


# ========================
# Data Fetching Utilities
# ========================

#get user transactions within a timeframe and add month column
def get_user_transactions(user_id: str, months: int = 12) -> pd.DataFrame:
    df = fetch_user_transactions(user_id)
    if df.empty:
        return df
    cutoff_date = datetime.now() - timedelta(days=months * 30)
    df = df[df["transaction_date"] >= cutoff_date]
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
    return df


def filter_by_type(df: pd.DataFrame, transaction_type: str) -> pd.DataFrame:
    """Filter transactions by type (income/expense)."""
    return df[df["type"] == transaction_type].copy()


# ========================
# Aggregation Utilities
# ========================

#get transactions summarized by month (income, expenses, net savings)
def summarize_monthly(df: pd.DataFrame) -> List[Dict[str, Any]]:
    summary = []
    for month in sorted(df["month"].unique()):
        mdf = df[df["month"] == month]

        # Handle both type conventions
        income = (
            mdf[mdf["type"].isin(["income"])]["transaction_amount"].sum()
        )
        expenses = (
            mdf[mdf["type"].isin(["expense"])]["transaction_amount"].sum()
        )

        summary.append({
            "month": month,
            "income": round(float(income), 2),
            "expenses": round(float(expenses), 2),
            "net_savings": round(float(income - expenses), 2)
        })
    return summary

#group transactions by category and calculate totals
def aggregate_by_category(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    category_summary = (
        df.groupby("category")["transaction_amount"]
        .agg(["sum", "count"])
        .reset_index()
    )
    category_summary.columns = ["category", "total_amount", "transaction_count"]
    category_summary = category_summary.sort_values("total_amount", ascending=False).head(limit)

    # Calculate percentages
    total_spending = category_summary["total_amount"].sum()
    if total_spending > 0:
        category_summary["percentage"] = (
            category_summary["total_amount"] / total_spending * 100
        ).round(2)
    else:
        category_summary["percentage"] = 0

    return category_summary


# ========================
# Formatting Utilities
# ========================

#format chart data as JSON for frontend
def chart_json(chart_type: str, labels: List, datasets: List[Dict], metadata: Dict) -> str:
    return json.dumps({
        "status": "success",
        "chart_type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "metadata": metadata
    }, indent=2)

#format error response as JSON
def error_json(error_message: str) -> str:
    return json.dumps({
        "status": "error",
        "error": error_message
    }, indent=2)

#format empty data response as JSON
def empty_response_json(chart_type: str = None, message: str = "No data found") -> str:
    response = {
        "status": "success",
        "data": [],
        "message": message
    }
    if chart_type:
        response["chart_type"] = chart_type
    return json.dumps(response, indent=2)


# ========================
# Transaction Formatting
# ========================
#format a single transaction record as a dictionary
def format_transaction_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "date": row["transaction_date"].strftime("%Y-%m-%d") if pd.notnull(row["transaction_date"]) else None,
        "description": str(row["description"]),
        "merchant": str(row["merchant"]) if "merchant" in row and pd.notnull(row["merchant"]) else None,
        "category": str(row["category"]) if "category" in row and pd.notnull(row["category"]) else None,
        "amount": round(float(row["transaction_amount"]), 2),
        "type": str(row["type"]),
        "account": str(row["account_name"]) if "account_name" in row and pd.notnull(row["account_name"]) else None
    }

def format_transactions_list(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    # drop duplicates
    df = df.drop_duplicates(subset=["id", "transaction_date", "transaction_amount"])
    
    return [
        format_transaction_record(row)
        for _, row in df.head(limit).iterrows()
    ]

# ========================
# Category Breakdown Utilities
# ========================

#format category summary as a list of detailed breakdowns
def format_category_breakdown(category_summary: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "category": row["category"],
            "amount": round(float(row["total_amount"]), 2),
            "percentage": round(float(row["percentage"]), 2),
            "transaction_count": int(row["transaction_count"])
        }
        for _, row in category_summary.iterrows()
    ]


# ========================
# Statistical Utilities
# ========================

#calculate common statistics for a list of numerical values
def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0
        }

    return {
        "mean": round(float(np.mean(values)), 2),
        "median": round(float(np.median(values)), 2),
        "std": round(float(np.std(values)), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "sum": round(float(np.sum(values)), 2)
    }
