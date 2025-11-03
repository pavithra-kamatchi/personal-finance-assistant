from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.tools.retriever_tool import fetch_user_transactions
import json


# ========================
# Data Fetching Utilities
# ========================

def get_user_transactions(user_id: str, months: int = 12) -> pd.DataFrame:
    """
    Fetch user transactions within a specified timeframe.
    Automatically adds 'month' column for aggregations.
    """
    df = fetch_user_transactions(user_id)
    if df.empty:
        return df
    cutoff_date = datetime.now() - timedelta(days=months * 30)
    df = df[df["transaction_date"] >= cutoff_date]
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
    return df


def filter_by_type(df: pd.DataFrame, transaction_type: str) -> pd.DataFrame:
    """Filter transactions by type (income/expense/debit/credit)."""
    return df[df["type"] == transaction_type].copy()


# ========================
# Aggregation Utilities
# ========================

def summarize_monthly(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Aggregate transactions by month, calculating income, expenses, and savings.
    Handles both 'income/expense' and 'credit/debit' type conventions.
    """
    summary = []
    for month in sorted(df["month"].unique()):
        mdf = df[df["month"] == month]

        # Handle both type conventions
        income = (
            mdf[mdf["type"].isin(["income", "credit"])]["transaction_amount"].sum()
        )
        expenses = (
            mdf[mdf["type"].isin(["expense", "debit"])]["transaction_amount"].sum()
        )

        summary.append({
            "month": month,
            "income": round(float(income), 2),
            "expenses": round(float(expenses), 2),
            "net_savings": round(float(income - expenses), 2)
        })
    return summary


def aggregate_by_category(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Group transactions by category and calculate totals.
    Returns top N categories by spending amount.
    """
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

def chart_json(chart_type: str, labels: List, datasets: List[Dict], metadata: Dict) -> str:
    """Format chart data as JSON for frontend consumption."""
    return json.dumps({
        "status": "success",
        "chart_type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "metadata": metadata
    }, indent=2)


def error_json(error_message: str) -> str:
    """Format error response as JSON."""
    return json.dumps({
        "status": "error",
        "error": error_message
    }, indent=2)


def empty_response_json(chart_type: str = None, message: str = "No data found") -> str:
    """Format empty data response as JSON."""
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

def format_transaction_record(row: pd.Series) -> Dict[str, Any]:
    """Format a single transaction row as a dictionary."""
    return {
        "id": row["id"],
        "date": row["transaction_date"].strftime("%Y-%m-%d"),
        "description": row["description"],
        "merchant": row.get("merchant"),
        "category": row.get("category"),
        "amount": round(float(row["transaction_amount"]), 2),
        "type": row["type"],
        "account": row.get("account_name")
    }


def format_transactions_list(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """Format multiple transactions as a list of dictionaries."""
    return [
        format_transaction_record(row)
        for _, row in df.head(limit).iterrows()
    ]


# ========================
# Category Breakdown Utilities
# ========================

def format_category_breakdown(category_summary: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format category summary as a list of detailed breakdowns."""
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

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate common statistical measures for a list of values."""
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
