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

#agent that classifies transactions and validates user input
class TransactionClassifierAgent:
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
        "type": "income", "expense", or null
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

    #process uploaded transactions with error handling
    def process_uploaded_transactions(self, inputs: List[dict], user_id: str) -> Optional[List[TransactionDetails]]:
        logger.info(f"Processing {len(inputs)} uploaded transactions")
        results = []
        errors = []

        # Check if inputs is empty
        if not inputs:
            logger.warning("No transactions provided to process")
            return []

        # Process each transaction with robust error handling
        for idx, transaction in enumerate(inputs):
            try:
                # Validate transaction has required fields
                if not isinstance(transaction, dict):
                    logger.error(f"Transaction {idx} is not a dictionary: {type(transaction)}")
                    errors.append(f"Row {idx}: Invalid format")
                    continue

                if "description" not in transaction or not transaction["description"]:
                    logger.error(f"Transaction {idx} missing 'description' field")
                    errors.append(f"Row {idx}: Missing description")
                    continue

                # Make sure it's a valid transaction
                logger.info(f"Validating transaction {idx + 1}/{len(inputs)}")
                try:
                    validation = self.transaction_validation(transaction)
                except Exception as e:
                    logger.error(f"Validation error for transaction {idx}: {e}")
                    errors.append(f"Row {idx}: Validation failed - {str(e)}")
                    continue

                # Check validation results
                if validation.confidence_score < 0.7 or not validation.is_transaction:
                    logger.warning(
                        f"Transaction {idx} failed gate check - "
                        f"is_transaction: {validation.is_transaction}, "
                        f"confidence: {validation.confidence_score:.2f}"
                    )
                    errors.append(
                        f"Row {idx}: Not a valid transaction "
                        f"(confidence: {validation.confidence_score:.2f})"
                    )
                    continue

                # Classify the transaction
                logger.info(f"Gate check passed for transaction {idx}, proceeding with classification")
                try:
                    transaction_info = self.LLMTransactionClassifierTool(transaction)

                    # Ensure it's a TransactionDetails object
                    if not isinstance(transaction_info, TransactionDetails):
                        transaction_info = TransactionDetails(**transaction_info)

                    # Add user_id
                    transaction_info = transaction_info.model_copy(update={"user_id": user_id})

                    logger.info(f"Transaction {idx} processed successfully: {transaction_info.description[:50]}...")
                    results.append(transaction_info)

                except Exception as e:
                    logger.error(f"Classification error for transaction {idx}: {e}")
                    errors.append(f"Row {idx}: Classification failed - {str(e)}")
                    continue

            except Exception as e:
                logger.error(f"Unexpected error processing transaction {idx}: {e}")
                errors.append(f"Row {idx}: Unexpected error - {str(e)}")
                continue

        # Log the summary
        logger.info(
            f"Transaction processing complete: "
            f"{len(results)} succeeded, {len(errors)} failed out of {len(inputs)} total"
        )

        if errors:
            logger.warning(f"Errors encountered: {errors[:5]}")  # Log first 5 errors

        return results if results else []