from pydantic import BaseModel, Field
from typing import List
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
    type: Optional[str] = Field(default=None, description= "The type of transaction: 'income' for money earned, 'expense' for money spent.")
    
class AuthInput(BaseModel):
    email: str
    password: str


class Budget(BaseModel):
    id: Optional[str] = Field(default=None, description="supabase-generated id")
    user_id: str = Field(description="User ID to associate with Supabase Auth user")
    category: str = Field(description="Spending category (e.g., 'Groceries', 'Dining', 'Transportation')")
    monthly_limit: float = Field(description="Monthly budget limit for this category")
    period: Optional[str] = Field(default="monthly", description="Budget period (e.g., 'monthly', 'weekly', 'yearly')")
    created_at: Optional[str] = Field(default=None, description="Timestamp when budget was created")


class BudgetCategory(BaseModel):
    category: str
    limit: float

class BudgetRequest(BaseModel):
    total_budget: float
    categories: List[BudgetCategory]