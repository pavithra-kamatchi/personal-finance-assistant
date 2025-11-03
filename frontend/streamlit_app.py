import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000" 

if "token" not in st.session_state:
    st.session_state.token = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"

st.title("💰 Personal Finance Assistant")

# Session state to store auth token
if "token" not in st.session_state:
    st.session_state.token = None

# --- Auth Section ---
st.subheader("🔐 Login or Signup")
auth_mode = st.radio("Choose", ["login", "signup"], horizontal=True)
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button(auth_mode):
    route = "login" if auth_mode == "login" else "signup"
    res = requests.post(f"{API_BASE}/auth/{route}", json={"email": email, "password": password})
    if res.status_code == 200:
        if route == "login":
            st.session_state.token = res.json()["access_token"]
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.success("Signed up! Check your email to confirm before logging in.")
    else:
        try:
            st.error(res.json().get("detail", res.text))
        except Exception:
            st.error(res.text)

# --- Upload Section ---
if st.session_state.token:
    st.subheader("📤 Upload Transaction CSV")
    uploaded_file = st.file_uploader("Choose a CSV", type="csv")

    if uploaded_file and st.button("Classify & Upload Transactions"):
        files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
        print("Token in session:", st.session_state.token)
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res = requests.post(f"{API_BASE}/transactions/upload-transactions", files={"file": uploaded_file}, headers=headers)

        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            try:
                st.error(res.json().get("detail", res.text))
            except Exception:
                st.error(res.text)