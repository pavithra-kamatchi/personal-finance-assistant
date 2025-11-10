import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import inspect
from pydantic import BaseModel
from backend.api.models.schemas import TransactionDetails
import pandas as pd
from backend.utils.db_connector import engine
from langchain_community.utilities.sql_database import SQLDatabase
      

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

# Initialise LangChain with OpenAI
openai_llm = OpenAI(api_key=openai_api_key)

#
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

# Define a simple prompt template for converting text to SQL
prompt_template = """
You are a SQL generator. Given the DB schema, the application model schema, a short conversation history (if any), and a natural language question, produce a single, syntactically correct {dialect} SQL SELECT query that answers the question.

Rules:
- ONLY produce a single SELECT query. Do NOT produce INSERT/UPDATE/DELETE/DROP/ALTER.
- Use only the tables and columns listed in the schema.
- Keep the query as simple as possible and include LIMIT when returning many rows.
- **Always filter results to only include rows where user_id = {user_id}.**
- **When generating SQL, always use the actual value of user_id provided above, not the string {user_id}.**

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
"""

# Create a PromptTemplate object
prompt = PromptTemplate(template=prompt_template, input_variables=["dialect", "table_info", "model_schema", "conversation", "user_id", "few_shot_examples", "input"])

# Create a chain
chain = (
    prompt 
    | openai_llm 
    | StrOutputParser()
)

def text_to_sql(
    question,
    user_id,
    user_conv=None,
    db=db,
    model=TransactionDetails,
    dialect="postgres"
):
    if user_conv is None:
        user_conv = []
    table_info = db.get_table_info()
    model_schema = model_schema_to_str(model)
    few_shot = FEW_SHOT_TEMPLATE.format(user_id=user_id)
    conversation = "\n".join([f"User: {q}" for q in user_conv[-6:][::-1]]) if user_conv else ""
    inputs = {
        "dialect": dialect,
        "table_info": table_info,
        "model_schema": model_schema,
        "conversation": conversation,
        "user_id": user_id,
        "few_shot_examples": few_shot,
        "input": question,
    }
    sql_query = chain.invoke(inputs)
    print(f"Generated SQL query: {sql_query}")
    try:
        df = pd.read_sql(sql_query, db._engine)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    # quick local test -- replace with a real UUID from your DB
        from backend.utils.sql_utils import run_validated_sql_agent 
        test_user = "123e4567-e89b-12d3-a456-426614174000"
        test_query = "How much did I spend on food last month?"

        try:
        # prefer validated_sql_execute (does validation + execution)
            result = run_validated_sql_agent(question=test_query, user_session_id=test_user, user_conv=[])
            print("Generated SQL:", result["sql"])
            print("Results (records):", result["results"].to_dict(orient="records"))
        except Exception as e:
        # fallback: use simple text_to_sql to inspect generated SQL without validation
            print("validated_sql_execute failed:", e)
        try:
            out = text_to_sql(test_query, test_user, user_conv=[])
            print("text_to_sql output:", out)
        except Exception as e2:
            print("text_to_sql also failed:", e2)
''' 
def run_sql_query(sql_query):
    with engine.connect() as conn:
        df = pd.read_sql(sql_query, conn)
    return df

# Define a function to convert text to SQL and execute the query
def text_to_sql(text):
    # Use LangChain to generate SQL from text
    sql_query = chain.invoke(text)
    
    print(f"Generated SQL query: {sql_query}")  # Print the generated query for debugging
    # Execute the SQL query and return results
    try:
        df = run_sql_query(sql_query)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

# Example prompts to test
prompts = [
    "Show all unique user_ids in the activity table",
]

# Convert the prompts to SQL and execute the queries
for prompt_text in prompts:
    print(f"\nExecuting prompt: {prompt_text}")
    results = text_to_sql(prompt_text)
    print(f"Results: {results}")
'''

#-------------------------

import pandas as pd
from backend.utils.db_connector import engine
from backend.agents.sql_nl_agent import nl2sql_agent
from typing import Tuple
from sqlalchemy import text
from langchain.schema import BaseMessage

FORBIDDEN_SQL = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE"]

import json

def extract_sql_from_agent_response(response):
    for msg in response.get("messages", []):
        # Try tool_calls attribute (as in your AIMessage)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                # If 'args' is a dict (as in your example)
                args = call.get("args")
                if isinstance(args, dict) and "query" in args:
                    return args["query"]
                # If 'arguments' is a JSON string (sometimes happens)
                arguments = call.get("function", {}).get("arguments")
                if arguments:
                    try:
                        args_dict = json.loads(arguments)
                        if "query" in args_dict:
                            return args_dict["query"]
                    except Exception:
                        pass
    return ""
'''
def extract_sql_from_agent_response(response) -> str:
    last_sql = ""

    # If it's just a single AIMessage, wrap it into a list for processing
    messages = []
    if isinstance(response, BaseMessage):
        messages = [response]
    elif isinstance(response, dict) and "messages" in response:
        messages = response["messages"]

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", []) or []
        for call in tool_calls:
            # Skip non-query tools
            if call.get("name") not in ("query_sql_db", "run_sql"):
                continue

            args = call.get("args")
            if isinstance(args, dict) and "query" in args:
                last_sql = args["query"].strip().rstrip(";")
            else:
                arguments = call.get("function", {}).get("arguments")
                if arguments:
                    try:
                        args_dict = json.loads(arguments)
                        if "query" in args_dict:
                            last_sql = args_dict["query"].strip().rstrip(";")
                    except Exception:
                        pass

    return last_sql
'''
def validate_sql(sql_query: str, allowed_tables=["transactions"]) -> Tuple[bool, str]:
    """Basic validation for destructive SQL or referencing allowed tables only."""
    upper_sql = sql_query.upper()
    for kw in FORBIDDEN_SQL:
        if kw in upper_sql:
            return False, f"Forbidden keyword found: {kw}"
    if not any(table.upper() in upper_sql for table in allowed_tables):
        return False, f"SQL does not reference allowed tables: {allowed_tables}"
    return True, "Validation passed"

def run_validated_sql_agent(user_question: str, user_session_id: str, max_attempts=4):
    """Invoke the NL2SQL agent with memory, validate and retry if needed, then execute."""
    attempt = 0
    last_error = None

    while attempt < max_attempts:
        # Invoke the agent with thread_id = user_session_id for memory persistence
        response = nl2sql_agent.invoke(
            {"input": user_question},
            config={"configurable": {"thread_id": user_session_id}},
        )
        print("Agent response:", response)
        for msg in response.get("messages", []):
            print("Message:", msg)
            print("Tool calls:", getattr(msg, "tool_calls", None))
            print("Content:", getattr(msg, "content", None))
        
        # Extract the SQL query from the response
        raw_sql = extract_sql_from_agent_response(response)
        print(f"Attempt {attempt + 1}: Generated SQL: {raw_sql}")
        if not raw_sql:
            last_error = "No SQL query found in agent response."
            attempt += 1
            continue
        
        # Validate SQL
        is_valid, message = validate_sql(raw_sql)
        if not is_valid:
            last_error = f"Validation failed: {message}"
            # You can optionally fix or retry or just raise here
            attempt += 1
            continue
        
        # Try executing safely
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(raw_sql), conn)
            return {
                "sql": raw_sql,
                "results": df,
                "validation": message,
                "attempts": attempt + 1,
            }
        except Exception as e:
            last_error = f"Execution error: {e}"
            attempt += 1

    raise RuntimeError(last_error or "Failed to produce valid and executable SQL")




#------------------------------------------------------------------------


#CSV_parser.py:

import pandas as pd
from typing import Optional, List
import io

DESCRIPTION_KEYS = ["description", "name", "details", "memo", "transaction_name", "transaction_description", "note", "transaction_note", "transaction_details"]

# Function to determine the description field in a CSV row
def infer_description_field(row: dict) -> Optional[str]:
    for key in row:
        if key.lower() in DESCRIPTION_KEYS:
            return key
    # Fallback: use longest string field as the best guess for the description
    string_val_dict = {k: v for k, v in row.items() if isinstance(v, str)}
    if not string_val_dict:
        return None
    return max(string_val_dict.items(), key=lambda item: len(item[1]))[0]

# Function to parse CSV file and infer the description field
def parse_csv_bytes(file_bytes: bytes) -> List[dict]:
    try:
        csv_string = file_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
        data = df.to_dict(orient='records')
        print("------------------------------------------")
        print("CSV data:", data)
        print("------------------------------------------")
        for row in data:
            print("Row Type:", type(row))
            description_key = infer_description_field(row)
            if description_key and description_key != "description":
                row["description"] = row[description_key]
                del row[description_key]
        print("Processed data:", data)
        print("------------------------------------------")
        return data

    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
        return []
    except pd.errors.ParserError:
        print("Error parsing CSV file. Please check the format.")
        return []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
#-----------------------------------------------------------------------

#Transaction_classifier.py:
from backend.api.models.schemas import TransactionCheck, TransactionDetails
from typing import Optional, List
from datetime import date
import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_together import Together
from langchain_openai import ChatOpenAI
from backend.tools.category_tool import fallback_category 

#start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#loading the .env file
load_dotenv()

# API Key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# LLM setup (LLaMA 3 70B)
tgt_llm = Together(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key=TOGETHER_API_KEY,
    temperature=0.3,
    max_tokens=512
)

# LLM setup (OpenAI GPT-4)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    max_tokens=512,
    openai_api_key=OPENAI_API_KEY,
)

# Output parsers
check_parser = JsonOutputParser(pydantic_object=TransactionCheck)
details_parser = JsonOutputParser(pydantic_object=TransactionDetails)

class TransactionClassifierAgent:
    """
    This class contains methods to classify transactions and validate user input.
    It uses LLMs to determine if a text describes a transaction and to classify the transaction details.
    """

    def __init__(self):
        self.tgt_llm = tgt_llm
        self.openai_llm = openai_llm
        self.check_parser = check_parser
        self.details_parser = details_parser
    
    #classify the transaction using LLM
    def LLMTransactionClassifierTool(self, user_input: dict) -> TransactionDetails:
        logger.info("Starting Transaction Classification")
        logger.debug(f"Input text: {user_input}")

        today = date.today()
        date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

        #bind the tools to the LLM
        logger.info("Binding tools to the LLM")
        llm_with_tools = self.openai_llm.bind_tools(
            tools=[fallback_category]
        )
        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{date_context} Your task is to determine the category of the transaction 
         such as 'Food', 'Entertainment', 'Groceries', etc based on the description of the transaction 
         and the merchant. If the merchant is provided in the transaction description, then extract the 
         merchant (e.g. Starbucks, Coldstone, Nike, etc.). 
         Extract the date in IOS format (YYYY-MM-DD) from the description if it is not provided.
        Respond with ONLY valid JSON. Do not include any extra text, explanation, or code block. 
        Format:
        {{{{
        "transaction_date": "date",
        "description": "string",
        "transaction_amount": "float",
        "category": "string",
        "merchant": "string",
        "account_name": "checking", "savings", or null
        "type": "debit", "credit", or null
        }}}}"""),
        ("user", "{text}")
        ])
        # Compose chain
        chain = prompt | llm_with_tools | details_parser

        # Run the chain
        logger.info("Running the LLM transaction classification chain")
        result: TransactionDetails = chain.invoke({"text": user_input["description"]})
        logger.info(f"Transaction classification complete: {result}")
        return result
        
    #obtain a confidence score and check whether the user input is a valid transaction
    def transaction_validation(self, user_input: dict) -> TransactionCheck:
        logger.info("Starting Transaction Validation analysis")
        logger.debug(f"Input text: {user_input}")

        today = date.today()
        date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."
        logger.info(f"Date context for validation: {date_context}")

        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{date_context} Your task is to determine whether the input describes a financial transaction based on the description.
        Respond with ONLY valid JSON. Do not include any extra text, explanation, or code block. 
        Format:
        {{{{
        "description": "string",
        "is_transaction": true or false,
        "confidence_score": float between 0 and 1
        }}}}"""),
        ("user", "{text}")
        ])

        # Compose chain
        logging.info("Composing the LLM chain for transaction validation")
        chain = prompt | self.openai_llm | check_parser
        logger.info("Chain composed successfully")

        # Run the chain
        result = chain.invoke({"text": user_input["description"]})
        if isinstance(result, dict):
            result = TransactionCheck(**result)
        logger.info(f"Extraction complete - Is transaction: {result.is_transaction}, Confidence: {result.confidence_score:.2f}")
        return result

    def process_uploaded_transactions(self, inputs: List[dict], user_id: str) -> Optional[List[TransactionDetails]]:
        #Implementing the prompt chain with gate check
        logger.info("Processing the uploaded transaction")
        results = []

        #check if the user inputted a valid transaction
        for transaction in inputs:
            validation = self.transaction_validation(transaction)
            #if validation fails, skip the transaction
            if validation.confidence_score < 0.7 or validation.is_transaction == False:
                logger.warning(
            f"Gate check failed - is_transaction: {validation.is_transaction}, confidence: {validation.confidence_score:.2f}"
            )
                logger.info("Gate check failed, skipping transaction processing")
                continue
            #if validation passes, classify the transaction
            else:
                logger.info("Gate check passed, proceeding with transaction processing")
                transaction_info = self.LLMTransactionClassifierTool(transaction)
                if not isinstance(transaction_info, TransactionDetails):
                    transaction_info = TransactionDetails(**transaction_info)
                transaction_info = transaction_info.model_copy(update={"user_id": user_id})
                logger.info(f"Transaction processed: {transaction_info}")
                results.append(transaction_info)
        return results if results else []
    
#------------------------------------------------------------------------
#Analytics agent with budgeting
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
    generate_insights_tool,
    budget_comparison_tool
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
    generate_insights_tool,
    budget_comparison_tool
]

# Bind tools to the LLM
llm_with_tools = openai_llm.bind_tools(tools)


# ========================
# LangGraph State and Nodes
# ========================

class AnalyticsAgentState(MessagesState):
    """Extended state for the Analytics agent with memory and budget."""
    user_id: str
    budget_data: Optional[Dict[str, float]] = None
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
    budget_data: Optional[Dict[str, float]] = None,
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
                                 "top_categories", "recent_transactions", "anomalies", "budget_comparison"]
        budget_data: Optional budget data as {"category": amount}
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
        if budget_data:
            analytics_types.append("budget_comparison")

    # Create system message with instructions
    system_message = f"""You are a financial analytics assistant. Your job is to help users understand their spending patterns and financial health.

User ID: {user_id}
Months to analyze: {months}
Budget provided: {"Yes" if budget_data else "No"}

You have access to the following analytics tools:
1. monthly_summary_tool - Calculate total income, expenses, and net savings per month
2. spending_over_time_tool - Get spending trends over time for line charts
3. income_vs_expense_tool - Compare income vs expenses for dual charts
4. top_categories_tool - Get top spending categories for pie/bar charts
5. recent_transactions_tool - Get the most recent 10 transactions
6. anomaly_detection_tool - Detect unusual/anomalous transactions
7. generate_insights_tool - Generate AI insights for any analytics data
8. budget_comparison_tool - Compare actual spending against budget (requires budget_data)

Your task is to generate the following analytics: {', '.join(analytics_types)}

For each analytics type:
1. Call the appropriate tool with the user_id
2. After getting the data, call generate_insights_tool to create meaningful insights
3. Ensure all data is in JSON format for the frontend

Important:
- Always pass the user_id to tools that require it
- For time-based tools, use months={months}
- For budget comparison, pass budget_json as JSON string: {json.dumps(budget_data) if budget_data else "N/A"}
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
                "budget_data": budget_data,
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
        "budget_comparison_tool": "budget_comparison",
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

def get_dashboard_analytics(user_id: str, budget_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Get all analytics needed for a dashboard view.
    This is a convenience function that requests all analytics types.

    Args:
        user_id: User ID to fetch transactions for
        budget_data: Optional budget data for comparison

    Returns:
        Complete analytics package for dashboard
    """
    return generate_analytics(
        user_id=user_id,
        analytics_types=None,  # All analytics
        budget_data=budget_data,
        months=6
    )


def get_specific_analytics(
    user_id: str,
    analytics_type: str,
    budget_data: Optional[Dict[str, float]] = None,
    months: int = 6
) -> Dict[str, Any]:
    """
    Get a specific type of analytics with insights.

    Args:
        user_id: User ID to fetch transactions for
        analytics_type: Single analytics type to generate
        budget_data: Optional budget data
        months: Number of months to analyze

    Returns:
        Specific analytics results with insights
    """
    return generate_analytics(
        user_id=user_id,
        analytics_types=[analytics_type],
        budget_data=budget_data,
        months=months
    )


# ========================
# Testing
# ========================

if __name__ == "__main__":
    # Test the analytics agent
    test_user = "123e4567-e89b-12d3-a456-426614174000"
    test_budget = {
        "Groceries": 500.0,
        "Dining": 300.0,
        "Transportation": 200.0,
        "Entertainment": 150.0,
        "Utilities": 250.0
    }

    print("Testing Analytics Agent")

    try:
        # Test getting all analytics
        result = get_dashboard_analytics(user_id=test_user, budget_data=test_budget)

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
#--------------------------------------------------------------------------------
#Analytics agent but without agent and just using backend sequential calls:
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=openai_api_key
)


# ========================
# Main Analytics Function
# ========================

def generate_dashboard_analytics(user_id: str, months: int = 6) -> Dict[str, Any]:
    """
    Fetch all analytics data first, then make a single LLM call to generate insights.
    Returns analytics data and insights in one response.
    """

    # Step 1: Fetch all analytics data
    all_data = {
        "monthly_summary": monthly_summary_tool(user_id),
        "spending_trends": spending_over_time_tool(user_id),
        "income_vs_expense": income_vs_expense_tool(user_id),
        "top_categories": top_categories_tool(user_id),
        "recent_transactions": recent_transactions_tool(user_id),
        "anomalies": anomaly_detection_tool(user_id)
    }

    # Step 2: Call LLM once to generate insights for all charts
    system_message = SystemMessage(
        content="You are a financial analytics assistant. Generate insights for each chart "
                "based on the provided analytics data."
    )

    user_message = HumanMessage(
        content=f"Here is the user's analytics data (months={months}):\n"
                f"{json.dumps(all_data, indent=2)}\n\n"
                "For each chart, generate meaningful insights explaining the data. "
                "Return a JSON object with keys matching the chart names and values as the insights."
    )

    try:
        response = llm.invoke([system_message, user_message])
        insights_json = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if LLM output is not perfectly formatted
        insights_json = {"error": "LLM response could not be parsed as JSON", "raw_response": response.content}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Step 3: Combine analytics data + insights
    combined_results = {}
    for chart_name, data in all_data.items():
        combined_results[chart_name] = {
            "data": data,
            "insights": insights_json.get(chart_name, "")
        }

    return {
        "status": "success",
        "user_id": user_id,
        "analytics": combined_results
    }


# ========================
# Convenience Functions
# ========================

def get_dashboard_analytics(user_id: str) -> Dict[str, Any]:
    """Convenience function to fetch dashboard analytics."""
    return generate_dashboard_analytics(user_id=user_id)


def get_specific_analytics(user_id: str, chart_name: str) -> Dict[str, Any]:
    """Fetch a specific chart analytics with insights."""
    result = generate_dashboard_analytics(user_id=user_id)
    chart_data = result["analytics"].get(chart_name, {})
    return chart_data


# ========================
# Testing
# ========================

if __name__ == "__main__":
    test_user = "123e4567-e89b-12d3-a456-426614174000"
    print("Testing Dashboard Analytics")
    result = get_dashboard_analytics(test_user)
    print(json.dumps(result, indent=2))














'''

import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import inspect
import re
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from backend.api.models.schemas import TransactionDetails
from backend.utils.db_connector import engine
from typing import Tuple
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser


# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Initialize the OpenAI chat model
openai_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.0,
    max_tokens=512,
    openai_api_key=OPENAI_API_KEY,
)

# Connect to the database
#db = SQLDatabase.from_uri(SUPABASE_DB_URL, include_tables=["transactions"])
engine = engine
"""
#Adding in MemorySaver to save the memory of the agent
memory_saver = MemorySaver()

#initialize the SQLDatabaseToolkit
sql_toolkit = SQLDatabaseToolkit(db = db, llm = openai_llm)
tools = sql_toolkit.get_tools()

#create the template for the agent
prompt = hub.pull("langchain-ai/sql-agent-system-prompt")

added_prompt = ("When answering, always produce a single call to the `query_sql_db` tool with a `query` field containing a valid SQL statement (only SELECT Statements allowed). Do not answer in plain text." \
"Few shot examples of valid SQL queries generated from natural language queries:\n" \
    "1. Natural Language query: 'Show me all transactions from 2023.'\n" \
       "SQL query: `SELECT * FROM transactions WHERE date >= '2023-01-01' AND date <= '2023-12-31'`\n" \
    "2. Natural Language query: 'What are the top 5 categories by transaction amount?'\n" \
       "SQL query: `SELECT category, SUM(amount) FROM transactions GROUP BY category ORDER BY SUM(amount) DESC LIMIT 5`\n" \
)
new_prompt = prompt.messages[0] + "\n" + added_prompt
# Create the React agent with the SQL database toolkit, prompt and memory saver
system_message = new_prompt.format(dialect = "postgresql", top_k = 10)
print("System message:", system_message)
nl2sql_agent = create_react_agent(
    openai_llm,
    tools=tools,
    prompt=system_message,
    checkpointer=memory_saver,
)
"""

PROMPT_TEMPLATE = """
You are a SQL generator. Given the DB schema, the application model schema, a short conversation history (if any), and a natural language question, produce a single, syntactically correct {dialect} SQL SELECT query that answers the question.

Rules:
- ONLY produce a single SELECT query. Do NOT produce INSERT/UPDATE/DELETE/DROP/ALTER.
- Use only the tables and columns listed in the schema.
- Keep the query as simple as possible and include LIMIT when returning many rows.
- **Always filter results to only include rows where user_id = '{user_id}'.**

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

FEW_SHOT = """
Question: How much did I spend on food last month?
SQLQuery: SELECT SUM(transaction_amount) FROM transactions WHERE category = 'Food' AND user_id = user_id AND transaction_date >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month') AND transaction_date < date_trunc('month', CURRENT_DATE);

Question: Show total spending by category in the last 3 months.
SQLQuery: SELECT category, SUM(transaction_amount) as total FROM transactions WHERE user_id = user_id AND transaction_date >= date_trunc('month', CURRENT_DATE - INTERVAL '3 months') GROUP BY category ORDER BY total DESC;
"""

# -------------------------
# Helpers: DB
# -------------------------


def get_table_info_from_engine(engine, table_name: str = "transactions") -> str:
    """Return a Postgres-style table description string for the prompt."""
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return f"{table_name} (table not found in DB)"

    cols = inspector.get_columns(table_name)
    col_defs = ", ".join([f"{c['name']} {c['type']}" for c in cols])
    create_stmt = f"{table_name}({col_defs})"
    return create_stmt


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

# -------------------------
# Validation
# -------------------------

def simple_sql_validation(engine, sql_query: str, allowed_table: str = "transactions") -> Tuple[bool, str]:
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE"]
    upper = sql_query.upper()
    for kw in forbidden:
        if kw in upper:
            return False, f"Destructive SQL keyword found: {kw}"

    # ensure it references the allowed table
    if allowed_table.lower() not in sql_query.lower():
        return False, f"Query does not reference the allowed table '{allowed_table}'"

    # Optional: check columns exist
    inspector = inspect(engine)
    cols = {c['name'].lower() for c in inspector.get_columns(allowed_table)}
    # find identifiers in SQL (naive)
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sql_query)
    referenced = {t.lower() for t in tokens}
    # check if at least one known column referenced
    if not (cols & referenced):
        return False, "No known columns referenced in query."

    return True, "Validation passed"

# -------------------------
# NL2SQL chain
# -------------------------

def build_sql_chain(llm=None) -> LLMChain:
    if llm is None:
        llm = openai_llm
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["table_info", "model_schema", "conversation", "input", "dialect", "few_shot_examples"]
    )
    chain = prompt | llm | StrOutputParser()
    return chain


def extract_sql_from_llm(raw: str) -> str:
    m = re.search(r"SQLQuery:\s*(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        sql_text = m.group(1).strip()
    else:
        sql_text = raw.strip()
    # Keep only first line or up to a terminating semicolon
    sql_text = sql_text.split("")[0]
    if ";" in sql_text:
        sql_text = sql_text.split(";")[-1] if sql_text.strip().endswith(";") else sql_text.split(";")[0]
    return sql_text


def validated_sql_execute(chain = build_sql_chain(openai_llm), engine = engine, question: str, user_session_id: str, user_conv: list, max_attempts: int = 2):
    """Use the LLMChain to generate SQL, validate it, attempt corrections if needed, execute and return results."""
    attempt = 0
    last_error = None
    table_info = get_table_info_from_engine(engine)
    model_schema = model_schema_to_str(TransactionDetails)
    while attempt < max_attempts:
        conversation_text = "".join([f"User: {q}" for q in user_conv[-6:][::-1]]) if user_conv else ""
        inputs = {
        "table_info": table_info,
        "model_schema": model_schema,
        "conversation": conversation_text,
        "input": question,
        "dialect": "postgres",
        "few_shot_examples": FEW_SHOT,
        "user_id": user_session_id,  # <-- add this
        }       
        raw = chain.invoke(inputs)
        raw_str = StrOutputParser().parse(raw)
        sql_query = extract_sql_from_llm(raw_str)

        # Basic normalization: ensure ; at end not included
        sql_query = sql_query.strip().rstrip(';')

        # Validate
        is_valid, msg = simple_sql_validation(engine, sql_query)
        if not is_valid:
            last_error = f"Validation failed: {msg} -- SQL: {sql_query}"
            # ask the model to correct given the error
            correction_input = question + "The previous generated SQL failed validation for this reason: " + msg + "Please produce a corrected SQLQuery using only the schema provided. SQLQuery:"
            raw = chain.invoke({**inputs, "input": correction_input})
            
            sql_query = extract_sql_from_llm(raw)
            attempt += 1
            continue

        # Ensure it starts with SELECT
        if not sql_query.strip().lower().startswith("select"):
            last_error = f"Only SELECT queries allowed. Generated: {sql_query}"
            correction_input = question + "Please produce a SELECT-only SQLQuery using only the schema provided. \n\n + SQLQuery:"
            raw = chain.run({**inputs, "input": correction_input})
            sql_query = extract_sql_from_llm(raw)
            attempt += 1
            continue

        # Execute
        try:
            with engine.connect() as conn:
                # Use text() to run raw SQL safely
                df = pd.read_sql(sql_query, conn)
            return {
                "sql": sql_query,
                "results": df,
                "validation": msg,
            }
        except Exception as e:
            last_error = f"Execution error: {e} -- SQL: {sql_query}"
            # ask LLM to fix using DB error
            correction_input = question + "The previous generated SQL failed to execute with error: " + str(e) + "Please produce a corrected SQLQuery. SQLQuery:"
            raw = chain.run({**inputs, "input": correction_input})
            sql_query = extract_sql_from_llm(raw)
            attempt += 1
            continue

    raise RuntimeError(last_error or "Failed to produce valid SQL")
'''



