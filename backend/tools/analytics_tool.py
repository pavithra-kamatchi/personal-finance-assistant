import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
from backend.tools.retriever_tool import fetch_user_transactions
from backend.utils.analytics_utils import (
    get_user_transactions,
    summarize_monthly,
    aggregate_by_category,
    chart_json,
    error_json,
    empty_response_json,
    format_transactions_list,
    format_category_breakdown
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Analytics tools for financial data analysis and visualization

@tool
def monthly_summary_tool(user_id: str, months: int = 12) -> str:
    """Returns monthly summary of income, expenses, and savings"""
    try:
        df = get_user_transactions(user_id, months)

        if df.empty:
            return empty_response_json(message="No transactions found")

        monthly_data = summarize_monthly(df)

        # Compute metadata
        incomes = [m["income"] for m in monthly_data]
        expenses = [m["expenses"] for m in monthly_data]
        savings = [m["net_savings"] for m in monthly_data]

        metadata = {
            "total_months": len(monthly_data),
            "avg_income": round(float(np.mean(incomes)), 2) if incomes else 0,
            "avg_expenses": round(float(np.mean(expenses)), 2) if expenses else 0,
            "avg_savings": round(float(np.mean(savings)), 2) if savings else 0
        }

        return json.dumps({
            "status": "success",
            "data": monthly_data,
            "metadata": metadata
        }, indent=2)

    except Exception as e:
        return error_json(str(e))

#get spending trends over time for line chart visualization
@tool
def spending_over_time_tool(user_id: str, months: int = 12) -> str:
    """Returns spending trends over time"""
    try:
        df = get_user_transactions(user_id, months)

        if df.empty:
            return empty_response_json(chart_type="line")

        # Filter expenses only
        expense_df = df[df["type"] == "expense"].copy()

        if expense_df.empty:
            return empty_response_json(chart_type="line", message="No expense transactions found")

        # Calculate monthly expenses
        monthly_expenses = (
            expense_df.groupby("month")["transaction_amount"]
            .sum()
            .reset_index()
            .sort_values("month")
        )
        monthly_expenses.columns = ["month", "amount"]

        # Prepare chart data
        labels = monthly_expenses["month"].tolist()
        amounts = [round(float(x), 2) for x in monthly_expenses["amount"].tolist()]

        datasets = [{
            "label": "Monthly Expenses",
            "data": amounts
        }]

        # Calculate metadata
        metadata = {
            "total_spending": round(float(sum(amounts)), 2),
            "avg_monthly_spending": round(float(np.mean(amounts)), 2),
            "max_month": monthly_expenses.loc[monthly_expenses["amount"].idxmax(), "month"] if not monthly_expenses.empty else None,
            "max_amount": round(float(max(amounts)), 2) if amounts else 0,
            "min_amount": round(float(min(amounts)), 2) if amounts else 0
        }

        return chart_json("line", labels, datasets, metadata)

    except Exception as e:
        return error_json(str(e))

#compare income vs expenses over time for dual line/bar chart
@tool
def income_vs_expense_tool(user_id: str, months: int = 6) -> str:
    """Returns income vs expenses comparison over time"""
    try:
        df = get_user_transactions(user_id, months)

        if df.empty:
            return empty_response_json(chart_type="bar")

        # Use the summarize_monthly helper
        monthly_summary = summarize_monthly(df)

        # Extract data for chart
        labels = [item["month"] for item in monthly_summary]
        income_data = [item["income"] for item in monthly_summary]
        expense_data = [item["expenses"] for item in monthly_summary]

        datasets = [
            {
                "label": "Income",
                "data": income_data,
                "type": "bar"
            },
            {
                "label": "Expenses",
                "data": expense_data,
                "type": "bar"
            }
        ]

        # Calculate metadata
        metadata = {
            "total_income": round(float(sum(income_data)), 2),
            "total_expenses": round(float(sum(expense_data)), 2),
            "net_savings": round(float(sum(income_data) - sum(expense_data)), 2),
            "avg_monthly_income": round(float(np.mean(income_data)), 2) if income_data else 0,
            "avg_monthly_expenses": round(float(np.mean(expense_data)), 2) if expense_data else 0
        }

        return chart_json("bar", labels, datasets, metadata)

    except Exception as e:
        return error_json(str(e))

#get the top spending categories for pie chart or bar chart visualization
@tool
def top_categories_tool(user_id: str, limit: int = 10, months: int = 12) -> str:
    """Returns top spending categories"""
    try:
        df = get_user_transactions(user_id, months)

        if df.empty:
            return empty_response_json(chart_type="pie")

        # Filter expenses only
        expense_df = df[df["type"] == "expense"].copy()

        if expense_df.empty:
            return empty_response_json(chart_type="pie", message="No expense transactions found")

        # Use helper function to aggregate by category
        category_summary = aggregate_by_category(expense_df, limit)

        # Prepare chart data
        labels = category_summary["category"].tolist()
        amounts = [round(float(x), 2) for x in category_summary["total_amount"].tolist()]

        datasets = [{
            "label": "Spending by Category",
            "data": amounts
        }]

        # Format breakdown using helper
        category_details = format_category_breakdown(category_summary)

        # Metadata
        total_spending = category_summary["total_amount"].sum()
        metadata = {
            "total_spending": round(float(total_spending), 2),
            "total_categories": len(category_summary),
            "top_category": category_summary.iloc[0]["category"] if not category_summary.empty else None,
            "top_category_amount": round(float(category_summary.iloc[0]["total_amount"]), 2) if not category_summary.empty else 0
        }

        return json.dumps({
            "status": "success",
            "chart_type": "pie",
            "data": {"labels": labels, "datasets": datasets},
            "breakdown": category_details,
            "metadata": metadata
        }, indent=2)

    except Exception as e:
        return error_json(str(e))

#get the most recent transactions for table display
@tool
def recent_transactions_tool(user_id: str, limit: int = 10) -> str:
    """Returns the 10 most recent transactions for a user"""
    try:
        # Fetch all transactions (no time limit for recent) 
        # get rid of this and just use pandas to get first 10 isntead of fetching the entire df
        df = fetch_user_transactions(user_id)

        if df.empty:
            return empty_response_json(message="No transactions found")

        # Use helper function to format transactions
        transactions = format_transactions_list(df, limit)

        return json.dumps({
            "status": "success",
            "data": transactions,
            "metadata": {
                "total_transactions": len(transactions)
            }
        }, indent=2)

    except Exception as e:
        return error_json(str(e))

#generate AI-powered insights based on analytics data
@tool
def generate_insights_tool(data_json: str, chart_type: str) -> str:
    """Returns AI-generated insights based on analytics data"""
    try:
        data = json.loads(data_json)

        # Create prompt based on chart type
        prompts = {
            "monthly_summary": f"""
Analyze this monthly financial summary and provide 3-4 key insights:
{json.dumps(data, indent=2)}

Focus on:
- Overall financial health (savings rate)
- Trends in income and expenses
- Notable months
- Recommendations for improvement
""",
            "spending_trends": f"""
Analyze these spending trends over time and provide 3-4 key insights:
{json.dumps(data, indent=2)}

Focus on:
- Spending patterns and trends
- Any concerning increases
- Seasonal variations
- Areas for potential savings
""",
            "income_vs_expense": f"""
Analyze this income vs expense comparison and provide 3-4 key insights:
{json.dumps(data, indent=2)}

Focus on:
- Income stability
- Expense patterns
- Savings potential
- Financial balance
""",
            "top_categories": f"""
Analyze this category spending breakdown and provide 3-4 key insights:
{json.dumps(data, indent=2)}

Focus on:
- Highest spending categories
- Percentage of total spending
- Opportunities to reduce costs
- Unusual spending patterns
""",
            "anomalies": f"""
Analyze these anomalous transactions and provide 3-4 key insights:
{json.dumps(data, indent=2)}

Focus on:
- Why these transactions are unusual
- Potential concerns or fraud indicators
- Context for high amounts
- Whether they're legitimate or need review
"""
        }

        prompt = prompts.get(chart_type, f"Analyze this financial data and provide key insights:\n{json.dumps(data, indent=2)}")

        # Generate insights using LLM
        response = llm.invoke(prompt)
        insights_text = response.content

        return json.dumps({
            "status": "success",
            "insights": insights_text,
            "chart_type": chart_type
        }, indent=2)

    except Exception as e:
        return error_json(str(e))

#compare actual spending against budgeted amounts
@tool
def budget_comparison_tool(user_id: str, budget_json: str) -> str:
    """Returns comparison of actual spending against budgeted amounts"""
    try:
        budget = json.loads(budget_json)

        # Get current month transactions only
        df = get_user_transactions(user_id, months=1)

        if df.empty:
            return empty_response_json(message="No transactions found for current month")

        # Get current month
        current_month = datetime.now().strftime("%Y-%m")

        # Filter expenses for current month
        current_month_df = df[
            (df["month"] == current_month) &
            (df["type"] == "expense")
        ]

        # Calculate spending by category
        actual_spending = current_month_df.groupby("category")["transaction_amount"].sum().to_dict()

        # Compare with budget
        comparison = []
        total_budget = 0
        total_spent = 0

        for category, budget_amount in budget.items():
            actual_amount = actual_spending.get(category, 0)
            difference = budget_amount - actual_amount
            percentage_used = (actual_amount / budget_amount * 100) if budget_amount > 0 else 0

            total_budget += budget_amount
            total_spent += actual_amount

            comparison.append({
                "category": category,
                "budget": round(float(budget_amount), 2),
                "actual": round(float(actual_amount), 2),
                "difference": round(float(difference), 2),
                "percentage_used": round(float(percentage_used), 2),
                "status": "over" if actual_amount > budget_amount else "under"
            })

        # Sort by percentage used (descending)
        comparison.sort(key=lambda x: x["percentage_used"], reverse=True)

        metadata = {
            "total_budget": round(float(total_budget), 2),
            "total_spent": round(float(total_spent), 2),
            "remaining_budget": round(float(total_budget - total_spent), 2),
            "overall_percentage_used": round(float(total_spent / total_budget * 100), 2) if total_budget > 0 else 0,
            "month": current_month,
            "categories_over_budget": sum(1 for c in comparison if c["status"] == "over")
        }

        return json.dumps({
            "status": "success",
            "data": comparison,
            "metadata": metadata
        }, indent=2)

    except Exception as e:
        return error_json(str(e))
