from wsgiref import headers
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional
import jwt
import os
from  supabase import create_client, Client
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED
from dotenv import load_dotenv
import requests
from jwt import decode, get_unverified_header, PyJWTError
from jwt.algorithms import RSAAlgorithm
from backend.api.models.schemas import AuthInput

auth_router = APIRouter()
load_dotenv()

# Load environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY").strip()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to handle user signup
@auth_router.post("/signup")
def signup(payload: AuthInput):
    try:
        response = supabase.auth.sign_up({
            "email": payload.email,
            "password": payload.password
        })
        # Check if response has 'user' or 'data'
        if not getattr(response, "user", None) and not getattr(response, "data", None):
            # Try to get error message from response
            error_msg = getattr(response, "message", "Unknown error")
            raise HTTPException(status_code=400, detail=error_msg)
        return {"message": "Signup successful. Please verify your email."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Function to handle user login
@auth_router.post("/login")
def login(payload: AuthInput):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })
        # Check if response has 'session'
        session = getattr(response, "session", None)
        if not session or not getattr(session, "access_token", None):
            # Try to get error message from response
            error_msg = getattr(response, "message", "Login failed.")
            raise HTTPException(status_code=401, detail=error_msg)
        access_token = session.access_token
        refresh_token = session.refresh_token
        return {"access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to get the current user based on the JWT token

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    # Extract the token from the header
    token = authorization.split(" ")[1]
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": SUPABASE_KEY
    }
    resp = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)
    # Check if the response is successful
    if resp.status_code != 200:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token verification failed")
    user_info = resp.json()
    return user_info["id"]

# Function to check if the user is authenticated
@auth_router.get("/auth-check")
def auth_check(user_id=Depends(get_current_user)):
    return {"message": "Authenticated", "user_id": user_id}
