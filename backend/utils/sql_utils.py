import pandas as pd
from backend.utils.db_connector import engine
from typing import Tuple, Dict, Any
from sqlalchemy import text, inspect
from langchain_core.tools import tool
import json
import re

FORBIDDEN_SQL = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE"]

# ========================
# SQL Validation Tools
# ========================

@tool
def validate_sql_query(sql_query: str) -> str:
    """
    SQLValidationTool: Validates a SQL query for safety and correctness.
    Checks for forbidden keywords and ensures it's a SELECT query.

    Args:
        sql_query: The SQL query to validate

    Returns:
        Validation result message (VALID or INVALID with reason)
    """
    upper = sql_query.upper()

    # Check for forbidden keywords
    for kw in FORBIDDEN_SQL:
        if kw in upper:
            return f"INVALID: Destructive SQL keyword found: {kw}. Only SELECT queries are allowed."

    # Ensure it's a SELECT query
    if not sql_query.strip().upper().startswith("SELECT"):
        return "INVALID: Only SELECT queries are allowed."

    # Check if references the transactions table
    if "transactions" not in sql_query.lower():
        return "INVALID: Query must reference the 'transactions' table."

    # Check for user_id filter (security check)
    if "user_id" not in sql_query.lower():
        return "INVALID: Query must include user_id filter for security."

    return "VALID: Query passed all validation checks."


def validate_sql(sql_query: str, allowed_tables=["transactions"]) -> Tuple[bool, str]:
    """
    Legacy validation function for backward compatibility.
    Basic validation for destructive SQL or referencing allowed tables only.
    """
    upper_sql = sql_query.upper()
    for kw in FORBIDDEN_SQL:
        if kw in upper_sql:
            return False, f"Forbidden keyword found: {kw}"
    if not any(table.upper() in upper_sql for table in allowed_tables):
        return False, f"SQL does not reference allowed tables: {allowed_tables}"
    return True, "Validation passed"


# ========================
# SQL Execution Tools
# ========================

@tool
def execute_sql_query(sql_query: str) -> str:
    """
    SQLExecutionTool: Executes a validated SQL query and returns the results.
    Only executes SELECT queries on the transactions table.

    Args:
        sql_query: The SQL query to execute

    Returns:
        Query results as a formatted string or error message
    """
    try:
        # Final safety check
        if not sql_query.strip().upper().startswith("SELECT"):
            return "ERROR: Only SELECT queries can be executed."

        # Execute the query
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)

        if df.empty:
            return "No results found for this query."

        # Convert to dict
        results = df.to_dict(orient="records")
        return json.dumps({
            "status": "success",
            "row_count": len(results),
            "results": results
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


# ========================
# Result Formatting Tools
# ========================

@tool
def format_sql_results(results_json: str) -> str:
    """
    ResultFormatterTool: Formats SQL query results into human-readable text.

    Args:
        results_json: JSON string containing query results

    Returns:
        Human-readable formatted results
    """
    try:
        data = json.loads(results_json)

        if data.get("status") == "error":
            return f"Error: {data.get('error', 'Unknown error')}"

        if data.get("status") == "success":
            results = data.get("results", [])
            row_count = data.get("row_count", 0)

            if row_count == 0:
                return "No results found."

            # Format as a readable table-like structure
            output = [f"Found {row_count} result(s):\n"]

            for i, row in enumerate(results, 1):
                output.append(f"\nResult {i}:")
                for key, value in row.items():
                    output.append(f"  {key}: {value}")

            return "\n".join(output)

        return str(data)

    except json.JSONDecodeError:
        return f"Results: {results_json}"
    except Exception as e:
        return f"Error formatting results: {str(e)}"


# ========================
# Helper Functions for Agent Response Extraction
# ========================

def extract_sql_from_agent_response(response) -> str:
    """
    Extracts SQL query from agent response messages.
    Looks for SQL in tool calls or message content.

    Args:
        response: Agent response containing messages

    Returns:
        Extracted SQL query string
    """
    from langchain_core.messages import BaseMessage

    last_sql = ""

    # Handle different response formats
    messages = []
    if isinstance(response, BaseMessage):
        messages = [response]
    elif isinstance(response, dict) and "messages" in response:
        messages = response["messages"]

    for msg in messages:
        # Check tool calls
        tool_calls = getattr(msg, "tool_calls", []) or []
        for call in tool_calls:
            # Skip non-SQL generation tools
            if call.get("name") in ("generate_sql_query", "nl_to_sql"):
                args = call.get("args")
                if isinstance(args, dict) and "query" in args:
                    last_sql = args["query"].strip().rstrip(";")

        # Also check content for SQL queries
        content = getattr(msg, "content", "")
        if content and isinstance(content, str):
            # Look for SELECT statements
            if "SELECT" in content.upper():
                # Try to extract SQL query
                match = re.search(r'(SELECT\s+.*?(?:;|$))', content, re.IGNORECASE | re.DOTALL)
                if match:
                    last_sql = match.group(1).strip().rstrip(';')

    return last_sql


def extract_results_from_response(response) -> Dict[str, Any]:
    """
    Extracts execution results from agent response messages.

    Args:
        response: Agent response containing messages

    Returns:
        Dictionary with results or error information
    """
    from langchain_core.messages import ToolMessage

    messages = response.get("messages", []) if isinstance(response, dict) else []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            try:
                # Try to parse as JSON
                data = json.loads(content)
                if "results" in data or "error" in data:
                    return data
            except:
                # Return raw content if not JSON
                if "ERROR" in content or "results" in content.lower():
                    return {"content": content}

    return {}


# ========================
# Database Schema Helper
# ========================

@tool
def get_table_schema() -> str:
    """
    Returns the schema information for the transactions table.
    Useful for understanding available columns and data types.

    Returns:
        Database schema information
    """
    from langchain_community.utilities.sql_database import SQLDatabase

    db = SQLDatabase(engine)
    table_info = db.get_table_info()
    return f"Database Schema:\n{table_info}"
