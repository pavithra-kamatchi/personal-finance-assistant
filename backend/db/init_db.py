from backend.utils.db_connector import engine
from sqlalchemy import text
import os

# Initialize the database schema and policies
# This script should be run once to set up the database schema and policies
# It reads SQL commands from a file and executes them to create the necessary tables and policies.
def initialize_db():
    queries_path = os.path.join(os.path.dirname(__file__), "queries.sql")
    with engine.begin() as conn:
        with open(queries_path, "r") as f:
            sql = f.read()
            # Split on semicolon, strip whitespace, ignore empty statements
            statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
            for stmt in statements:
                conn.execute(text(stmt))

if __name__ == "__main__":
    initialize_db()
    print("Database schema and policies initialized.")