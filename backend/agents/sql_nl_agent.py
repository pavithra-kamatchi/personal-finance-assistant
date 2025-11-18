import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from sqlalchemy import inspect
from pydantic import BaseModel
from backend.api.models.schemas import TransactionDetails
import pandas as pd
from backend.utils.db_connector import engine
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import tools from sql_utils
from backend.utils.sql_utils import (
    validate_sql_query,
    execute_sql_query,
    format_sql_results,
    get_table_schema,
    extract_sql_from_agent_response,
    extract_results_from_response
)

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Ensure the API key is correctly loaded
if not openai_api_key:
    raise ValueError("API key for OpenAI not found. Please set it in the .env file.")

# Connect to your PostgreSQL database
db = SQLDatabase(engine)

# Initialize LangChain with OpenAI (using ChatOpenAI for LangGraph)
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=openai_api_key
)

# Helper function to convert Pydantic model to string
def model_schema_to_str(model: BaseModel) -> str:
    """Convert Pydantic model to simple field:type and description lines."""
    m = model.model_json_schema()
    props = m.get("properties", {})
    lines = []
    for name, info in props.items():
        t = info.get("type", "")
        desc = info.get("description", "")
        lines.append(f"{name}: {t} - {desc}")
    return "\n".join(lines)

# Define the prompt template (same as original)
PROMPT_TEMPLATE = """
You are a SQL generator. Given the DB schema, the application model schema, a short conversation history (if any), and a natural language question, produce a single, syntactically correct {dialect} SQL SELECT query that answers the question.

Rules:
- ONLY produce a single SELECT query. Do NOT produce INSERT/UPDATE/DELETE/DROP/ALTER.
- Use only the tables and columns listed in the schema.
- Keep the query as simple as possible and include LIMIT when returning many rows.
- **Always filter results to only include rows where user_id = {user_id}.**
- **When generating SQL, always use the actual value of user_id provided above, not the string {{user_id}}.**

Database Schema:
{table_info}

Application Model (fields & types):
{model_schema}

Conversation history (most recent first):
{conversation}

User ID: {user_id}

Few-shot examples:
{few_shot_examples}

Question: {input}

SQLQuery:
"""

FEW_SHOT_TEMPLATE = """
Question: How much did I spend on food last month?
SQLQuery: SELECT SUM(transaction_amount) FROM transactions WHERE category = 'Food' AND user_id = '{user_id}' AND transaction_date >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month') AND transaction_date < date_trunc('month', CURRENT_DATE);

Question: Show total spending by category in the last 3 months.
SQLQuery: SELECT category, SUM(transaction_amount) as total FROM transactions WHERE user_id = '{user_id}' AND transaction_date >= date_trunc('month', CURRENT_DATE - INTERVAL '3 months') GROUP BY category ORDER BY total DESC;

Question: Show my average transaction amount by merchant.
SQLQuery: SELECT merchant_name, AVG(transaction_amount) FROM transactions WHERE user_id = '{user_id}' GROUP BY merchant_name;

Question: How much is my budget and how much can I still spend?
SQLQuery: SELECT b.category, b.budget_amount, COALESCE(SUM(t.transaction_amount), 0) as spent, b.budget_amount - COALESCE(SUM(t.transaction_amount), 0) as remaining FROM budgets b LEFT JOIN transactions t ON b.category = t.category AND b.user_id = t.user_id WHERE b.user_id = '{user_id}' GROUP BY b.category, b.budget_amount;
"""

# Create the NL to SQL tool using the prompt template
@tool
def nl_to_sql(question: str, user_id: str, conversation: str = "") -> str:
    """
    NLtoSQLTool: Converts a natural language question into a SQL query.
    Uses the database schema, application model, and conversation history to generate accurate queries.

    Args:
        question: The natural language question to convert to SQL
        user_id: The user ID to filter results (for security)
        conversation: Previous conversation context (optional)

    Returns:
        A SQL SELECT query string
    """
    table_info = db.get_table_info()
    model_schema = model_schema_to_str(TransactionDetails)
    few_shot = FEW_SHOT_TEMPLATE.format(user_id=user_id)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["dialect", "table_info", "model_schema", "conversation", "user_id", "few_shot_examples", "input"]
    )

    chain = prompt | openai_llm | StrOutputParser()

    inputs = {
        "dialect": "postgres",
        "table_info": table_info,
        "model_schema": model_schema,
        "conversation": conversation,
        "user_id": user_id,
        "few_shot_examples": few_shot,
        "input": question,
    }

    sql_query = chain.invoke(inputs)

    # Extract SQL from response (in case LLM adds extra text)
    sql_query = sql_query.strip()
    if "SQLQuery:" in sql_query:
        sql_query = sql_query.split("SQLQuery:")[-1].strip()

    # Remove any trailing semicolons or extra whitespace
    sql_query = sql_query.rstrip(';').strip()

    return sql_query

# Create the tools list - all tools now defined
tools = [nl_to_sql, validate_sql_query, execute_sql_query, format_sql_results, get_table_schema]

# Bind tools to the LLM
llm_with_tools = openai_llm.bind_tools(tools)

# Define the agent state
class AgentState(MessagesState):
    """Extended state for the SQL agent with memory."""
    user_id: str

# Define the agent node
def call_model(state: AgentState):
    """Agent node that calls the LLM with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define a node to handle tool calls
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to continue with tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, continue to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "end"

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

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
nl2sql_agent = workflow.compile(checkpointer=memory)

# Main function to interact with the agent
def text_to_sql(
    question: str,
    user_id: str,
    user_conv: list = None,
    max_iterations: int = 15
):
    """
    Main function to convert natural language to SQL and execute.
    This orchestrates the entire workflow using the LangGraph agent.

    Args:
        question: Natural language question about transactions
        user_id: User ID for filtering and session management
        user_conv: Conversation history (optional)
        max_iterations: Maximum number of agent iterations

    Returns:
        Dict with SQL query, results, and response
    """
    if user_conv is None:
        user_conv = []

    # Format conversation history
    conversation_text = "\n".join([f"User: {q}" for q in user_conv[-6:][::-1]]) if user_conv else ""

    # Create system message with instructions
    system_message = f"""You are a helpful SQL assistant. Your job is to help users query their transaction data.

User ID: {user_id}

Follow these steps IN ORDER:
1. Use the 'nl_to_sql' tool to convert the user's question into a SQL query
2. Use the 'validate_sql_query' tool to validate the generated SQL
3. If valid, use the 'execute_sql_query' tool to run the query
4. After getting the query results, provide a clear, conversational summary directly to the user

IMPORTANT: After step 4, DO NOT call any more tools. Just provide your final answer and STOP.

If validation fails, try to fix the SQL and validate again (maximum 2 attempts).
Always ensure the query filters by user_id for security."""

    # Create the user message
    user_message = question
    if conversation_text:
        user_message = f"Previous conversation:\n{conversation_text}\n\nCurrent question: {question}"

    # Invoke the agent with memory
    config = {
        "configurable": {"thread_id": user_id},
        "recursion_limit": max_iterations
    }

    try:
        result = nl2sql_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_message)
                ],
                "user_id": user_id
            },
            config=config
        )

        # Extract information from the result
        sql_query = extract_sql_from_agent_response(result)
        results_data = extract_results_from_response(result)

        # Get the final AI response
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, "content") else str(final_message)

        return {
            "sql": sql_query,
            "results": results_data.get("results") if isinstance(results_data, dict) else None,
            "response": response_text,
            "messages": result["messages"]
        }

    except Exception as e:
        return {"error": str(e)}


# Legacy function for backward compatibility
def run_validated_sql_agent(question: str, user_session_id: str, user_conv: list = None):
    """
    Legacy wrapper function for backward compatibility.
    Calls the new text_to_sql function.
    """
    result = text_to_sql(question=question, user_id=user_session_id, user_conv=user_conv)

    if "error" in result:
        raise RuntimeError(result["error"])

    # Convert to legacy format
    if result.get("results"):
        df = pd.DataFrame(result["results"])
    else:
        df = pd.DataFrame()

    return {
        "sql": result.get("sql", ""),
        "results": df,
        "validation": "Validation passed",
        "attempts": 1
    }


if __name__ == "__main__":
    # Test the agent
    test_user = "123e4567-e89b-12d3-a456-426614174000"
    test_query = "How much did I spend on food last month?"

    print(f"Testing NL2SQL Agent with question: {test_query}\n")

    try:
        result = text_to_sql(question=test_query, user_id=test_user, user_conv=[])
        print("=" * 50)
        print("SQL Query Generated:")
        print(result.get("sql", "No SQL found"))
        print("\n" + "=" * 50)
        print("Results:")
        print(result.get("results", "No results"))
        print("\n" + "=" * 50)
        print("Agent Response:")
        print(result.get("response", "No response"))
    except Exception as e:
        print(f"Error: {e}")
