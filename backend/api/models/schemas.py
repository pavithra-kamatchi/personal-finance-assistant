from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

class TransactionCheck(BaseModel):
    description: str = Field(description = "raw description of the transaction")
    is_transaction: bool = Field(description = "whether this text describes a transaction")
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class TransactionDetails(BaseModel):
    id: Optional[str] = Field(default=None, description = "supabase-generated id")
    user_id: Optional[str] = Field(default=None, description = "to associate with Supabase Auth user")
    transaction_date: date = Field(description="The date when the transaction occurred.")
    description: str = Field(description="raw description of the transaction")
    transaction_amount: float = Field(description="payment amount in transaction")
    category: Optional[str] = Field(description="The predicted category of the transaction, such as 'Groceries', 'Utilities', 'Dining', etc., based on the transaction description and merchant.")
    merchant: Optional[str] = Field(default = None, description = "The merchant or vendor associated with the transaction, inferred from the transaction description (e.g., 'Amazon', 'Starbucks').")
    account_name: Optional[str] = Field(default=None, description= "The name of the account from which the transaction was made (e.g., 'Checking', 'Savings').")
    type: Optional[str] = Field(default=None, description= "The type of transaction, typically 'debit' or 'credit'.")
    
class AuthInput(BaseModel):
    email: str
    password: str