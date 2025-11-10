import os
import json
from dotenv import load_dotenv
from typing import Literal, Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Import all analytics tools
from backend.tools.analytics_tool import (
    monthly_summary_tool,
    spending_over_time_tool,
    income_vs_expense_tool,
    top_categories_tool,
    recent_transactions_tool,
    generate_insights_tool
)

from backend.tools.anomaly_tool import anomaly_detection_tool

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("API key for OpenAI not found. Please set it in the .env file.")

# Initialize LLM
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=openai_api_key
)

# Define all analytics tools
tools = [
    monthly_summary_tool,
    spending_over_time_tool,
    income_vs_expense_tool,
    top_categories_tool,
    recent_transactions_tool,
    anomaly_detection_tool,
    generate_insights_tool
]

# Bind tools to the LLM
llm_with_tools = openai_llm.bind_tools(tools)


# ========================
# LangGraph State and Nodes
# ========================

class AnalyticsAgentState(MessagesState):
    """Extended state for the Analytics agent with memory."""
    user_id: str
    analytics_results: Dict[str, Any] = {}


def call_model(state: AnalyticsAgentState):
    """Agent node that calls the LLM with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AnalyticsAgentState) -> Literal["tools", "end"]:
    """Determine whether to continue with tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, continue to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "end"


# ========================
# Build LangGraph Workflow
# ========================

workflow = StateGraph(AnalyticsAgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# Add memory for conversation persistence
memory = MemorySaver()

# Compile the graph with memory
analytics_agent = workflow.compile(checkpointer=memory)


# ========================
# Main Analytics Function
# ========================

def generate_analytics(
    user_id: str,
    analytics_types: Optional[List[str]] = None,
    months: int = 6,
    max_iterations: int = 20
) -> Dict[str, Any]:
    """
    Main function to generate comprehensive analytics for a user.
    Uses LangGraph agent to orchestrate multiple analytics tools.

    Args:
        user_id: User ID to fetch transactions for
        analytics_types: List of analytics to generate (if None, generates all)
                        Options: ["monthly_summary", "spending_trends", "income_vs_expense",
                                 "top_categories", "recent_transactions", "anomalies"]
        months: Number of months to analyze (default: 6)
        max_iterations: Maximum number of agent iterations (default: 20)

    Returns:
        Dictionary containing all analytics results with insights
    """
    # Default to all analytics if not specified
    if analytics_types is None:
        analytics_types = [
            "monthly_summary",
            "spending_trends",
            "income_vs_expense",
            "top_categories",
            "recent_transactions",
            "anomalies"
        ]

    # Create system message with instructions
    system_message = f"""You are a financial analytics assistant. Your job is to help users understand their spending patterns and financial health.

User ID: {user_id}
Months to analyze: {months}

You have access to the following analytics tools:
1. monthly_summary_tool - Calculate total income, expenses, and net savings per month
2. spending_over_time_tool - Get spending trends over time for line charts
3. income_vs_expense_tool - Compare income vs expenses for dual charts
4. top_categories_tool - Get top spending categories for pie/bar charts
5. recent_transactions_tool - Get the most recent 10 transactions
6. anomaly_detection_tool - Detect unusual/anomalous transactions
7. generate_insights_tool - Generate AI insights for any analytics data

Your task is to generate the following analytics: {', '.join(analytics_types)}

For each analytics type:
1. Call the appropriate tool with the user_id
2. After getting the data, call generate_insights_tool to create meaningful insights
3. Ensure all data is in JSON format for the frontend

Important:
- Always pass the user_id to tools that require it
- For time-based tools, use months={months}
- Generate insights for EACH chart/analytics type
- Be comprehensive and ensure all requested analytics are completed
"""

    # Create user message
    user_message = f"""Please generate comprehensive financial analytics for me.

Analytics requested: {', '.join(analytics_types)}

For each analytics:
1. Fetch the data using the appropriate tool
2. Generate AI insights explaining what the data means
3. Return everything in JSON format

Make sure to complete all analytics and provide insights for each."""

    # Invoke the agent with memory
    config = {
        "configurable": {"thread_id": user_id},
        "recursion_limit": max_iterations
    }

    try:
        result = analytics_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_message)
                ],
                "user_id": user_id,
                "analytics_results": {}
            },
            config=config
        )

        # Extract analytics results from tool messages
        analytics_results = extract_analytics_from_response(result)

        # Get the final AI response
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, "content") else str(final_message)

        return {
            "status": "success",
            "user_id": user_id,
            "analytics": analytics_results,
            "summary": response_text,
            "messages": result["messages"]
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def extract_analytics_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract analytics data from agent response messages.
    Parses tool outputs and organizes them by analytics type.

    Args:
        response: Agent response containing messages

    Returns:
        Dictionary with organized analytics results
    """
    from langchain_core.messages import ToolMessage

    messages = response.get("messages", [])
    analytics_results = {}

    # Map of tool names to analytics types
    tool_mapping = {
        "monthly_summary_tool": "monthly_summary",
        "spending_over_time_tool": "spending_trends",
        "income_vs_expense_tool": "income_vs_expense",
        "top_categories_tool": "top_categories",
        "recent_transactions_tool": "recent_transactions",
        "anomaly_detection_tool": "anomalies",
        "generate_insights_tool": "insights"
    }

    # Extract data from tool messages
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            tool_name = msg.name if hasattr(msg, "name") else None

            try:
                # Parse JSON content
                data = json.loads(content)

                # Determine analytics type
                analytics_type = tool_mapping.get(tool_name, "unknown")

                # Handle insights separately (they're linked to specific chart types)
                if analytics_type == "insights":
                    chart_type = data.get("chart_type")
                    if chart_type:
                        # Add insights to the corresponding analytics type
                        target_type = tool_mapping.get(f"{chart_type}_tool", chart_type)
                        if target_type in analytics_results:
                            analytics_results[target_type]["insights"] = data.get("insights", "")
                        else:
                            analytics_results[f"{chart_type}_insights"] = data.get("insights", "")
                else:
                    # Store analytics data
                    analytics_results[analytics_type] = data

            except json.JSONDecodeError:
                # If not JSON, store as raw content
                analytics_type = tool_mapping.get(tool_name, "unknown")
                analytics_results[analytics_type] = {"raw": content}

    return analytics_results


# ========================
# Convenience Functions
# ========================

def get_dashboard_analytics(user_id: str) -> Dict[str, Any]:
    """
    Get all analytics needed for a dashboard view.
    This is a convenience function that requests all analytics types.

    Args:
        user_id: User ID to fetch transactions for

    Returns:
        Complete analytics package for dashboard
    """
    return generate_analytics(
        user_id=user_id,
        analytics_types=None,  # All analytics
        months=6
    )


def get_specific_analytics(
    user_id: str,
    analytics_type: str,
    months: int = 6
) -> Dict[str, Any]:
    """
    Get a specific type of analytics with insights.

    Args:
        user_id: User ID to fetch transactions for
        analytics_type: Single analytics type to generate
        months: Number of months to analyze

    Returns:
        Specific analytics results with insights
    """
    return generate_analytics(
        user_id=user_id,
        analytics_types=[analytics_type],
        months=months
    )


# ========================
# Testing
# ========================

if __name__ == "__main__":
    # Test the analytics agent
    test_user = "123e4567-e89b-12d3-a456-426614174000"

    print("Testing Analytics Agent")

    try:
        # Test getting all analytics
        result = get_dashboard_analytics(user_id=test_user)

        print("\nStatus:", result.get("status"))
        print("\nAnalytics Generated:")
        for analytics_type, data in result.get("analytics", {}).items():
            print(f"\n  - {analytics_type}:")
            if isinstance(data, dict):
                print(f"    Status: {data.get('status', 'N/A')}")
                if "metadata" in data:
                    print(f"    Metadata: {data['metadata']}")

        print("\n" + "=" * 80)
        print("Summary:")
        print(result.get("summary", "No summary available"))
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
