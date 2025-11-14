import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI

# Import all analytics tools - we'll call them directly as functions
from backend.tools.analytics_tool import (
    monthly_summary_tool,
    spending_over_time_tool,
    income_vs_expense_tool,
    top_categories_tool,
    recent_transactions_tool,
    budget_comparison_tool
)

from backend.tools.anomaly_tool import anomaly_detection_tool

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("API key for OpenAI not found. Please set it in the .env file.")

# Initialize LLM for insights generation
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Safely run a tool and parse JSON output
def run_tool_safely(tool_func, **kwargs) -> Dict[str, Any]:
    try:
        result_str = tool_func.invoke(kwargs)
        return json.loads(result_str)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# run all tools concurrently by executing all tools in parallel and collect results and store in a dictionary 
def run_analytics_tools_concurrently(
    user_id: str,
    months: int = 6,
    budget_data: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    results = {}

    # Define all tool executions
    tool_tasks = {
        "monthly_summary": (monthly_summary_tool, {"user_id": user_id, "months": months}),
        "spending_trends": (spending_over_time_tool, {"user_id": user_id, "months": months}),
        "income_vs_expense": (income_vs_expense_tool, {"user_id": user_id, "months": months}),
        "top_categories": (top_categories_tool, {"user_id": user_id, "limit": 10, "months": months}),
        "recent_transactions": (recent_transactions_tool, {"user_id": user_id, "limit": 10}),
        "anomalies": (anomaly_detection_tool, {"user_id": user_id, "months": months})
    }

    # Add budget comparison if budget data is provided
    if budget_data:
        tool_tasks["budget_comparison"] = (
            budget_comparison_tool,
            {"user_id": user_id, "budget_json": json.dumps(budget_data)}
        )

    # Execute all tools concurrently
    with ThreadPoolExecutor(max_workers=len(tool_tasks)) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(run_tool_safely, tool_func, **kwargs): name
            for name, (tool_func, kwargs) in tool_tasks.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }

    return results


#Generate comprehensive insights from all analytics data in a single LLM call
def generate_comprehensive_insights(analytics_data: Dict[str, Any]) -> str:
    # Build comprehensive prompt with all analytics data
    prompt = f"""You are a financial analyst providing comprehensive insights about a user's financial data.

Below is the complete financial analytics data. Provide a comprehensive analysis covering ALL aspects:

=== MONTHLY SUMMARY ===
{json.dumps(analytics_data.get('monthly_summary', {}), indent=2)}

=== SPENDING TRENDS ===
{json.dumps(analytics_data.get('spending_trends', {}), indent=2)}

=== INCOME VS EXPENSE ===
{json.dumps(analytics_data.get('income_vs_expense', {}), indent=2)}

=== TOP SPENDING CATEGORIES ===
{json.dumps(analytics_data.get('top_categories', {}), indent=2)}

=== RECENT TRANSACTIONS ===
{json.dumps(analytics_data.get('recent_transactions', {}), indent=2)}

=== ANOMALOUS TRANSACTIONS ===
{json.dumps(analytics_data.get('anomalies', {}), indent=2)}

{"=== BUDGET COMPARISON ===" if analytics_data.get('budget_comparison') else ""}
{json.dumps(analytics_data.get('budget_comparison', {}), indent=2) if analytics_data.get('budget_comparison') else ""}

Please provide a comprehensive financial analysis with the following structure:

1. **Overall Financial Health**: Summarize the user's financial situation (income, expenses, savings rate)

2. **Spending Patterns**: Analyze spending trends, top categories, and any concerning patterns

3. **Budget Performance** (if applicable): How well they're staying within budget limits

4. **Anomalies & Alerts**: Highlight any unusual transactions or red flags

5. **Actionable Recommendations**: Provide 3-5 specific, actionable recommendations to improve their financial health

Keep insights clear, concise, and actionable. Use bullet points where appropriate."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"


#main function to generate dashboard analytics
def generate_dashboard_analytics(
    user_id: str,
    months: int = 6,
    budget_data: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate complete dashboard analytics with a single LLM call for insights.
    This is optimized for low latency by:
    1. Running all analytics tools concurrently
    2. Making only ONE LLM call to generate comprehensive insights

    Args:
        user_id: User ID to fetch transactions for
        months: Number of months to analyze (default: 6)
        budget_data: Optional budget data dict (e.g., {"Groceries": 500, "Dining": 300})

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - user_id: The user ID
        - analytics: All analytics data
        - insights: Comprehensive AI-generated insights
        - execution_time: Time taken to generate analytics
    """
    import time
    start_time = time.time()

    try:
        # Step 1: Run all analytics tools concurrently
        analytics_results = run_analytics_tools_concurrently(
            user_id=user_id,
            months=months,
            budget_data=budget_data
        )

        # Step 2: Generate comprehensive insights in a single LLM call
        comprehensive_insights = generate_comprehensive_insights(analytics_results)

        execution_time = time.time() - start_time

        return {
            "status": "success",
            "user_id": user_id,
            "analytics": analytics_results,
            "insights": comprehensive_insights,
            "execution_time": round(execution_time, 2),
            "metadata": {
                "months_analyzed": months,
                "has_budget_comparison": budget_data is not None,
                "analytics_count": len(analytics_results)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id
        }


#get specific analytics without comprehensive insights for individual chart updates
def get_specific_analytics(
    user_id: str,
    analytics_type: str,
    months: int = 6,
    budget_data: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Get a specific type of analytics (without comprehensive insights).
    Useful for individual chart updates.

    Args:
        user_id: User ID to fetch transactions for
        analytics_type: Type of analytics ("monthly_summary", "spending_trends", etc.)
        months: Number of months to analyze
        budget_data: Optional budget data for budget comparison

    Returns:
        Specific analytics result
    """
    tool_map = {
        "monthly_summary": (monthly_summary_tool, {"user_id": user_id, "months": months}),
        "spending_trends": (spending_over_time_tool, {"user_id": user_id, "months": months}),
        "income_vs_expense": (income_vs_expense_tool, {"user_id": user_id, "months": months}),
        "top_categories": (top_categories_tool, {"user_id": user_id, "limit": 10, "months": months}),
        "recent_transactions": (recent_transactions_tool, {"user_id": user_id, "limit": 10}),
        "anomalies": (anomaly_detection_tool, {"user_id": user_id, "months": months}),
        "budget_comparison": (budget_comparison_tool, {"user_id": user_id, "budget_json": json.dumps(budget_data or {})})
    }

    if analytics_type not in tool_map:
        return {
            "status": "error",
            "error": f"Unknown analytics type: {analytics_type}"
        }

    tool_func, kwargs = tool_map[analytics_type]
    return run_tool_safely(tool_func, **kwargs)


#testing the analytics agent
if __name__ == "__main__":
    # Test the analytics agent with the new concurrent approach
    test_user = "af34934d-a0ac-422e-9dc6-e15553635846"

    print("=" * 80)
    print("Testing Analytics Agent (LangChain - Concurrent Execution)")
    print("=" * 80)

    # Sample budget data
    test_budget = {
        "Groceries": 500.0,
        "Dining": 300.0,
        "Transportation": 200.0,
        "Entertainment": 150.0
    }

    try:
        # Test getting all analytics with budget
        print("\n[1] Generating dashboard analytics with budget comparison...")
        result = generate_dashboard_analytics(
            user_id=test_user,
            months=6,
            budget_data=test_budget
        )

        print(f"\nStatus: {result.get('status')}")
        print(f"Execution Time: {result.get('execution_time')}s")
        print(f"\nMetadata:")
        print(f"  - Months Analyzed: {result.get('metadata', {}).get('months_analyzed')}")
        print(f"  - Analytics Count: {result.get('metadata', {}).get('analytics_count')}")
        print(f"  - Has Budget: {result.get('metadata', {}).get('has_budget_comparison')}")

        print("\n" + "-" * 80)
        print("Analytics Generated:")
        print("-" * 80)
        for analytics_type, data in result.get("analytics", {}).items():
            if isinstance(data, dict):
                status = data.get("status", "N/A")
                print(f"\n  ✓ {analytics_type}: {status}")
                if "metadata" in data:
                    print(f"    Metadata: {json.dumps(data['metadata'], indent=6)}")
                if data.get("status") == "error":
                    print(f"    Error: {data.get('error')}")

        print("\n" + "=" * 80)
        print("COMPREHENSIVE INSIGHTS:")
        print("=" * 80)
        print(result.get("insights", "No insights available"))
        print("=" * 80)

        # Test getting specific analytics
        print("\n\n[2] Testing specific analytics retrieval...")
        specific_result = get_specific_analytics(
            user_id=test_user,
            analytics_type="top_categories",
            months=6
        )
        print(f"\nTop Categories Result:")
        print(json.dumps(specific_result, indent=2)[:500] + "...")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
