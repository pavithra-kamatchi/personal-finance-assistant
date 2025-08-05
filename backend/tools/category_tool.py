import re
from langchain_core.tools import tool
from pydantic import BaseModel

class CategoryInput(BaseModel):
    description: str

@tool(args_schema=CategoryInput)
def fallback_category(description: str) -> str:
    """
    Classifies a transaction description into a fallback category.
    """
    categories = {
        "groceries": [
            "whole foods", "trader joe", "grocery", "walmart", "costco", "safeway", "aldi", "kroger", "publix", "supermarket"
        ],
        "food_delivery": [
            "doordash", "grubhub", "ubereats", "postmates", "seamless", "caviar", "delivery"
        ],
        "restaurants": [
            "restaurant", "diner", "bistro", "cafe", "eatery", "chipotle", "mcdonald", "burger king", "subway", "panera", "starbucks"
        ],
        "entertainment": [
            "movie", "cinema", "concert", "netflix", "spotify", "hulu", "amc", "regal", "theater", "disney+", "paramount+", "eventbrite"
        ],
        "transportation": [
            "uber", "lyft", "taxi", "cab", "evgo", "bus", "train", "metro", "subway", "transit", "amtrak", "boltbus", "greyhound"
        ],
        "shopping": [
            "mall", "amazon", "nike", "clothes", "shoes", "target", "best buy", "apple", "electronics", "gap", "old navy", "uniqlo", "zara", "h&m"
        ],
        "online_shopping": [
            "amazon", "amzn", "ebay", "etsy", "shopify", "wish", "aliexpress", "online"
        ],
        "travel": [
            "airlines", "delta", "united", "american airlines", "jetblue", "southwest", "hotel", "marriott", "hilton", "airbnb", "expedia", "booking.com", "trip"
        ],
        "gas": [
            "chevron", "shell", "gas", "exxon", "bp", "mobil", "petrol", "fuel"
        ],
        "utilities": [
            "electric", "water", "gas bill", "utility", "comcast", "xfinity", "at&t", "verizon", "internet", "phone", "cellular"
        ],
        "health": [
            "pharmacy", "walgreens", "cvs", "rite aid", "doctor", "hospital", "clinic", "dentist", "vision", "health", "medication", "prescription"
        ],
        "fitness": [
            "gym", "planet fitness", "24 hour fitness", "la fitness", "yoga", "pilates", "workout", "fitness"
        ],
        "insurance": [
            "insurance", "geico", "state farm", "allstate", "progressive", "aetna", "blue cross", "cigna"
        ],
        "education": [
            "school", "college", "university", "tuition", "course", "udemy", "coursera", "edx", "textbook", "learning"
        ],
        "charity": [
            "donation", "charity", "gofundme", "red cross", "unicef", "foundation"
        ],
        "personal_care": [
            "salon", "spa", "barber", "haircut", "nails", "massage", "beauty"
        ],
        "pets": [
            "petco", "petsmart", "vet", "pet", "dog", "cat", "animal"
        ],
        "government": [
            "dmv", "tax", "irs", "passport", "license", "fee", "city", "state"
        ],
        "finance": [
            "bank", "atm", "transfer", "deposit", "withdrawal", "chase", "boa", "wells fargo", "credit card", "payment"
        ],
        "other": []
    }

    desc_lower = description.lower()
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return "other"