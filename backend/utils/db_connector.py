import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

# Load environment variables from .env file
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

#Create the database engine
engine = create_engine(SUPABASE_DB_URL, echo=True)
