# app.py

import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time
import numpy as np
import threading
import av  # Needed to return frames correctly
import warnings
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# Import functions from our custom modules
from face_registration import save_face_data
from take_attendance import load_model, mark_attendance

# --- Page Configuration ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📋 Smart Attendance System")

ATTENDANCE_DIR = "Attendance"

# --- Initialize Session State ---
if "start_registration" not in st.session_state:
    st.session_state.start_registration = False
if "captured_faces" not in st.session_state:
    st.session_state.captured_faces = []
if "new_name" not in st.session_state:
    st.session_state.new_name = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "recognized_name" not in st.session_state:
    st.session_state.recognized_name = "Unknown"

# --- Load Models and Data ---
try:
    facedetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading Haar Cascade file: {e}. Make sure the file is in the 'Data' directory.")
    st.stop()

knn, error_message = load_model()
if error_message:
    st.warning(error_message)

# --- Video Processor for Registration ---
class RegistrationProcessor(VideoTransformerBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.last_capture_time = 0
        self.frame_count = 0
        self.local_captures = []

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        with self.lock:
            current_time = time.time()
            if len(faces) > 0:
                st.session_state.feedback = "Face Detected!"
                if current_time - self.last_capture_time > 1.0:
                    if len(self.local_captures) < 5:
                        (x, y, w, h) = faces[0]
                        crop_img = img[y:y+h, x:x+w]
                        if crop_img.size > 0:
                            resized_img = cv2.resize(crop_img, (50, 50))
                            self.local_captures.append(resized_img)
                            self.last_capture_time = current_time
                            print(f"*** CAPTURED IMAGE #{len(self.local_captures)} ***")
            else:
                st.session_state.feedback = "No Face Detected"

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- App Sections ---

# Section 1: Register New Face  (UNCHANGED)
with st.container():
    st.subheader("🧑‍💻 Register New Face")
    st.session_state.new_name = st.text_input("Enter your name:", value=st.session_state.get('new_name', ''), key="new_name_input")

    if st.button("📸 Start Registration", key="start_reg_btn"):
        if not st.session_state.new_name:
            st.warning("Please enter a name before registering.")
        else:
            st.session_state.captured_faces = []
            st.session_state.start_registration = True
            st.rerun()

    if st.session_state.start_registration:
        if len(st.session_state.captured_faces) < 5:
            st.warning("Click the 'START' button below to turn on your camera.")

            ctx = webrtc_streamer(
                key="registration",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=RegistrationProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=False
            )

            if ctx.video_processor:
                with ctx.video_processor.lock:
                    st.session_state.captured_faces = ctx.video_processor.local_captures.copy()

            if ctx.state.playing:
                st.info("Please show your face to the camera. Capturing 5 images...")

            st.info(st.session_state.get('feedback', 'Initializing...'))
            st.progress(len(st.session_state.captured_faces) / 5)
            st.write(f"Captured: {len(st.session_state.captured_faces)}/5")

            if ctx.state.playing:
                time.sleep(0.2)
                st.rerun()

        else:
            st.session_state.start_registration = False
            st.success("Capture complete! Saving your face data...")
            st.balloons()

            success, message = save_face_data(st.session_state.new_name, st.session_state.captured_faces)
            if success:
                st.success(message)
                st.info("Data saved. Please refresh the page to update the attendance model.")
            else:
                st.error(message)

            st.session_state.captured_faces = []
            st.session_state.new_name = ""
            time.sleep(3)
            st.rerun()

# Section 2: Take Attendance  (UNCHANGED)
with st.container():
    st.subheader("✅ Take Attendance")
    attendance_register=set()
    if knn:
        st.info("Click 'START' below to begin attendance.")

        class AttendanceProcessor(VideoTransformerBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                recognized_name = ""

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    crop_img = img[y:y+h, x:x+w]
                    if crop_img.size > 0:
                        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                        output = knn.predict(resized_img)
                        recognized_name = output[0]
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)

                st.session_state["recognized_name"] = recognized_name
                if recognized_name and recognized_name not in attendance_register:
                    attendance_register.add(recognized_name)
                    message = mark_attendance(recognized_name)
                    st.info(message)  # Shows success/error in the UI
                    print(f"Message is {message}")
                    if "Error" in message:
                        st.error(message)
                    else:
                        st.success(message)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        ctx_att = webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=AttendanceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False
        )

        if ctx_att.state.playing:
            time.sleep(0.1)
            st.rerun()
    else:
        st.info("Please register a face before taking attendance.")

# ===== Section 3: Attendance Viewer & Analytics (ENHANCED, SAFE) =====
with st.container():
    st.subheader("📅 Attendance Viewer & Analytics")

    # ---- Select Date & Show That Day's Attendance ----
    today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    selected_date = st.date_input("Select date to view attendance", value=today_ist)

    date_str = selected_date.strftime("%d-%m-%Y")
    filename = f"{ATTENDANCE_DIR}/Attendance_{date_str}.csv"

    if os.path.exists(filename):
        try:
            df_day = pd.read_csv(filename)

            st.markdown("### 📄 Attendance for Selected Date")

            # Summary metrics
            total_entries = len(df_day)
            unique_students = df_day["NAME"].nunique() if "NAME" in df_day.columns else 0
            first_time = df_day["TIME"].min() if "TIME" in df_day.columns else "-"
            last_time = df_day["TIME"].max() if "TIME" in df_day.columns else "-"

            c1, c2 = st.columns(2)
            c1.metric("Total Entries", total_entries)
            c2.metric("Unique Students", unique_students)

            # Full time range as normal text (no truncation)
            st.write(f"⏳ **Time Range:** `{first_time}` — `{last_time}`")

            st.dataframe(df_day)

            # Download button for this day's CSV
            with open(filename, "rb") as f:
                st.download_button(
                    label="⬇ Download this day's CSV",
                    data=f,
                    file_name=os.path.basename(filename),
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not read the attendance file: {e}")
    else:
        st.warning("No attendance has been recorded for the selected date yet.")

    # ---- Analytics Across All Days ----
    st.markdown("### 📊 Attendance Analytics (All Days)")

    if os.path.exists(ATTENDANCE_DIR):
        frames = []
        for fname in os.listdir(ATTENDANCE_DIR):
            if fname.startswith("Attendance_") and fname.endswith(".csv"):
                fpath = os.path.join(ATTENDANCE_DIR, fname)
                try:
                    df_tmp = pd.read_csv(fpath)
                    # Extract date from filename: Attendance_DD-MM-YYYY.csv
                    date_part = fname.replace("Attendance_", "").replace(".csv", "")
                    df_tmp["DATE"] = date_part
                    frames.append(df_tmp)
                except Exception:
                    continue

        if frames:
            df_all = pd.concat(frames, ignore_index=True)

            if "NAME" in df_all.columns and "DATE" in df_all.columns:
                total_days = df_all["DATE"].nunique()
                summary = (
                    df_all.groupby("NAME")["DATE"]
                    .nunique()
                    .reset_index(name="Days Present")
                    .sort_values("Days Present", ascending=False)
                )

                if total_days > 0:
                    summary["Attendance %"] = (summary["Days Present"] / total_days * 100).round(2)

                st.dataframe(summary)

                # Simple bar chart: Days Present per student
                st.bar_chart(
                    data=summary.set_index("NAME")["Days Present"],
                    use_container_width=True
                )
            else:
                st.info("Attendance data is missing NAME/DATE columns.")
        else:
            st.info("No attendance files found yet for analytics.")
    else:
        st.info("Attendance directory does not exist yet. Take some attendance first.")
