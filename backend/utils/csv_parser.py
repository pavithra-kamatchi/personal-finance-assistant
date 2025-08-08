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
