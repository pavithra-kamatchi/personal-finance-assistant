import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import inspect
import re

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Initialize the OpenAI chat model
openai_llm = init_chat_model(
    model="gpt-4",
    temperature=0.0,
    max_tokens=512,
    openai_api_key=OPENAI_API_KEY,
)

# Connect to the database
db = SQLDatabase.from_uri(SUPABASE_DB_URL, include_tables=["transactions"])

#Adding in MemorySaver to save the memory of the agent
memory_saver = MemorySaver()

#initialize the SQLDatabaseToolkit
sql_toolkit = SQLDatabaseToolkit(db = db, llm = openai_llm)
tools = tools = sql_toolkit.get_tools()

#create the template for the agent
prompt = hub.pull("langchain-ai/sql-agent-system-prompt")
prompt.messages[0].pretty_print()

# Create the React agent with the SQL database toolkit, prompt and memory saver
system_message = prompt.format(dialect = 'postgres', top_k = 10)
nl2sql_agent = create_react_agent(
    openai_llm,
    tools=tools,
    prompt=system_message,
    checkpointer=memory_saver,
)


