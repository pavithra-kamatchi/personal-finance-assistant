import pandas as pd
from typing import List
import io
import logging

logger = logging.getLogger(__name__)

#clean row data by removing NaN values and converting to JSON-serializable format
def clean_row_data(row: dict) -> dict:
    cleaned = {}
    for key, value in row.items():
        # Skip NaN/None values
        if pd.isna(value) or value is None:
            continue

        # Convert numpy types to Python types
        if hasattr(value, 'item'):
            value = value.item()

        cleaned[key] = value

    return cleaned

#convert entire row to descriptive string that LLM can parse
def row_to_description(row: dict) -> str:
    cleaned_row = clean_row_data(row)

    if not cleaned_row:
        return ""

    # Format as a natural language description
    parts = []
    for key, value in cleaned_row.items():
        # Create key-value pairs in a readable format
        parts.append(f"{key}: {value}")

    # Join all parts with commas
    description = ", ".join(parts)

    return description

#validate that row has at least some data
def validate_row(row: dict) -> bool:
    cleaned = clean_row_data(row)
    return len(cleaned) > 0

#main function to parse CSV bytes into LLM-friendly format
def parse_csv_bytes(file_bytes: bytes) -> List[dict]:
    try:
        # Try UTF-8 encoding first
        csv_string = file_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))

        logger.info(f"CSV loaded with {len(df)} rows and columns: {df.columns.tolist()}")

    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        logger.warning("UTF-8 decode failed, trying latin-1 encoding")
        try:
            csv_string = file_bytes.decode('latin-1')
            df = pd.read_csv(io.StringIO(csv_string))
            logger.info(f"CSV loaded with latin-1 encoding: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to decode CSV with latin-1: {e}")
            return []

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty")
        return []

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        return []

    except Exception as e:
        logger.error(f"Unexpected error reading CSV: {e}")
        return []

    # Check if DataFrame is empty
    if df.empty:
        logger.warning("CSV file has no data rows")
        return []

    # Convert to list of dictionaries
    data = df.to_dict(orient='records')

    # Transform each row into LLM-friendly format
    processed_data = []
    for idx, row in enumerate(data):
        try:
            if not validate_row(row):
                logger.warning(f"Row {idx} has no valid data, skipping")
                continue

            # Convert entire row to a descriptive string
            description = row_to_description(row)

            if not description or len(description) < 5:
                logger.warning(f"Row {idx} produced empty or too short description, skipping")
                continue

            # Pass the entire row as description for the LLM to parse
            processed_row = {
                "description": description,
                "raw_data": clean_row_data(row)  # Keep raw data for reference
            }

            processed_data.append(processed_row)
            logger.debug(f"Row {idx} processed: {description[:100]}...")

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    logger.info(f"Successfully processed {len(processed_data)} out of {len(data)} rows")

    return processed_data
