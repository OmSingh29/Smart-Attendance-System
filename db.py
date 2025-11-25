# db.py

from pymongo import MongoClient
import streamlit as st

def get_mongo_client():
    """
    Returns a MongoClient instance if MONGO_URI is configured in Streamlit secrets.
    Otherwise returns None so the app can still run without cloud DB.
    """
    uri = None

    # Read from Streamlit secrets (works locally with .streamlit/secrets.toml
    # and on Streamlit Cloud with app secrets)
    try:
        if "MONGO_URI" in st.secrets:
            uri = st.secrets["MONGO_URI"]
    except Exception:
        uri = None

    if not uri:
        return None

    try:
        client = MongoClient(uri)
        return client
    except Exception as e:
        print(f"[Cloud DB] Failed to create MongoClient: {e}")
        return None


def get_attendance_collection():
    """
    Returns a MongoDB collection for storing attendance.
    If configuration is missing or connection fails, returns None.
    """
    client = get_mongo_client()
    if client is None:
        return None

    db_name = "smart_attendance"
    col_name = "attendance_records"

    try:
        if "MONGO_DB_NAME" in st.secrets:
            db_name = st.secrets["MONGO_DB_NAME"]
        if "MONGO_COLLECTION" in st.secrets:
            col_name = st.secrets["MONGO_COLLECTION"]
    except Exception:
        pass

    db = client[db_name]
    return db[col_name]
