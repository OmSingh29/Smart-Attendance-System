# take_attendance.py

import pickle
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

from db import get_attendance_collection  # <-- cloud DB helper

warnings.filterwarnings("ignore")


def load_model():
    """
    Loads face data and trains a KNN classifier.
    Returns a tuple: (classifier, error_message).
    """
    try:
        with open('Data/names.pkl', 'rb') as w:
            LABELS = pickle.load(w)
        with open('Data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)
        return knn, None
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        return None, f"Error loading model data: {e}. Please register a face first."


def mark_attendance(name):
    """
    Marks attendance for a given name by saving it to MongoDB.

    - One record per person per day.
    - Uses Asia/Kolkata timezone.
    """
    try:
        print("Entered mark_attendance")

        ts = datetime.now(ZoneInfo("Asia/Kolkata"))
        date_str = ts.strftime("%d-%m-%Y")
        time_str = ts.strftime("%H:%M:%S")

        collection = get_attendance_collection()
        if collection is None:
            # Hard stop because you explicitly want cloud DB as source of truth
            return "Cloud database is not configured. Please set MONGO_* in secrets."

        # Check if attendance already marked today
        existing = collection.find_one({"name": name, "date": date_str})
        if existing:
            print("Already marked in MongoDB")
            return "This person's attendance has already been taken"

        doc = {
            "name": name,
            "date": date_str,       # "DD-MM-YYYY"
            "time": time_str,       # "HH:MM:SS"
            "timestamp": ts.isoformat(),
            "source": "streamlit_app"
        }

        collection.insert_one(doc)
        print(f"[Cloud DB] Saved attendance to MongoDB for {name} on {date_str} at {time_str}")

        return f"Attendance marked for {name} at {time_str}"

    except Exception as e:
        return f"Error marking attendance: {e}"
