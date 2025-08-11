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