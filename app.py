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
from pathlib import Path

warnings.filterwarnings("ignore")

# Import functions from our custom modules
from face_registration import save_face_data
from take_attendance import load_model, mark_attendance

# --- Page Configuration ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("Smart Attendance System")

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
if "erase_all_confirm" not in st.session_state:
    st.session_state.erase_all_confirm = False
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "admin_password" not in st.session_state:
    st.session_state.admin_password = ""


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

# ----- Section 3: Simple Today View (optional, kept if you like) -----
with st.container():
    st.subheader("📅 Today's Attendance (Quick View)")
    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y")
    today_file = f"{ATTENDANCE_DIR}/Attendance_{today_str}.csv"
    if os.path.exists(today_file):
        try:
            df_today = pd.read_csv(today_file)
            st.dataframe(df_today)
        except Exception as e:
            st.error(f"Could not read today's attendance file: {e}")
    else:
        st.warning("No attendance has been recorded for today yet.")

# ===== Section 4: 🔐 Admin Panel (Edit + Per-Student Analytics + Comparison) =====
with st.container():
    st.subheader("🔐 Admin Panel")

    # If NOT in admin mode yet → show login
    if not st.session_state.is_admin:
        password_input = st.text_input(
            "Enter admin password",
            type="password",
            key="admin_password_input"
        )

        if st.button("🔓 Login as Admin"):
            if password_input == "admin123":
                st.session_state.is_admin = True
                st.session_state.admin_password = ""
                st.success("Admin mode enabled. Loading admin tools...")
                st.rerun()  # re-run so that the 'else' block below is executed
            else:
                st.error("Incorrect admin password.")
    else:
        # Already in admin mode
        st.success("✅ Admin mode enabled")

        # Logout button
        if st.button("🚪 Exit Admin Mode"):
            st.session_state.is_admin = False
            st.session_state.admin_password = ""
            st.session_state.erase_all_confirm = False
            st.info("You have exited admin mode.")
            st.rerun()

        tab1, tab2, tab3 = st.tabs(["✏️ Edit Attendance", "📊 Student Analytics", "⚖️ Compare Students"])

        # ---------- TAB 1: EDIT ATTENDANCE ----------
        with tab1:
            st.markdown("#### ✏️ Edit / Add / Delete Attendance Records")

            today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
            edit_date = st.date_input("Select date to edit", value=today_ist, key="admin_edit_date")

            edit_date_str = edit_date.strftime("%d-%m-%Y")
            edit_filename = f"{ATTENDANCE_DIR}/Attendance_{edit_date_str}.csv"

            if os.path.exists(edit_filename):
                df_edit = pd.read_csv(edit_filename)
            else:
                # Empty DataFrame with correct columns if file doesn't exist
                df_edit = pd.DataFrame(columns=["NAME", "TIME"])

            st.info("You can add new rows or modify/delete existing rows below.")
            edited_df = st.data_editor(
                df_edit,
                num_rows="dynamic",
                key="admin_editor"
            )

            if st.button("💾 Save changes for selected date"):
                # Ensure Attendance directory exists
                if not os.path.exists(ATTENDANCE_DIR):
                    os.makedirs(ATTENDANCE_DIR)
                edited_df.to_csv(edit_filename, index=False)
                st.success(f"Saved changes to {edit_filename}")

                st.caption(
                    "- To add attendance of a person who has not come: add a new row with NAME and TIME.\n"
                    "- To delete a record: remove that row from the table before saving."
                )

        # ---------- TAB 2: STUDENT ANALYTICS ----------
        with tab2:
            st.markdown("#### 📊 Analytics for a Single Student")

            all_frames = []
            if os.path.exists(ATTENDANCE_DIR):
                for fname in os.listdir(ATTENDANCE_DIR):
                    if fname.startswith("Attendance_") and fname.endswith(".csv"):
                        fpath = os.path.join(ATTENDANCE_DIR, fname)
                        try:
                            df_tmp = pd.read_csv(fpath)
                            date_part = fname.replace("Attendance_", "").replace(".csv", "")
                            df_tmp["DATE"] = date_part
                            all_frames.append(df_tmp)
                        except Exception:
                            continue

            if all_frames:
                df_all = pd.concat(all_frames, ignore_index=True)

                if "NAME" in df_all.columns and "DATE" in df_all.columns:
                    students = sorted(df_all["NAME"].dropna().unique())
                    selected_student = st.selectbox("Select student", students)

                    df_student = df_all[df_all["NAME"] == selected_student]

                    if not df_student.empty:
                        st.write(f"##### Attendance for: {selected_student}")

                        total_days = df_all["DATE"].nunique()
                        days_present = df_student["DATE"].nunique()

                        att_percent = (days_present / total_days * 100) if total_days > 0 else 0

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Days Present", days_present)
                        c2.metric("Total Days (in data)", total_days)
                        c3.metric("Attendance %", f"{att_percent:.2f}%")

                        st.markdown("**Dates Present:**")
                        st.write(sorted(df_student["DATE"].unique()))

                        st.markdown("**Detailed Records:**")
                        st.dataframe(df_student[["DATE", "TIME"]].sort_values("DATE"))
                    else:
                        st.info("No records found for this student.")
                else:
                    st.info("Attendance data is missing NAME/DATE columns.")
            else:
                st.info("No attendance data found yet.")

        # ---------- TAB 3: COMPARE STUDENTS ----------
        with tab3:
            st.markdown("#### ⚖️ Compare Students")

            all_frames_cmp = []
            if os.path.exists(ATTENDANCE_DIR):
                for fname in os.listdir(ATTENDANCE_DIR):
                    if fname.startswith("Attendance_") and fname.endswith(".csv"):
                        fpath = os.path.join(ATTENDANCE_DIR, fname)
                        try:
                            df_tmp = pd.read_csv(fpath)
                            date_part = fname.replace("Attendance_", "").replace(".csv", "")
                            df_tmp["DATE"] = date_part
                            all_frames_cmp.append(df_tmp)
                        except Exception:
                            continue

            if all_frames_cmp:
                df_all_cmp = pd.concat(all_frames_cmp, ignore_index=True)

                if "NAME" in df_all_cmp.columns and "DATE" in df_all_cmp.columns:
                    students_all = sorted(df_all_cmp["NAME"].dropna().unique())
                    selected_students = st.multiselect(
                        "Select students to compare",
                        students_all,
                        max_selections=5
                    )

                    if selected_students:
                        total_days = df_all_cmp["DATE"].nunique()

                        comp = (
                            df_all_cmp[df_all_cmp["NAME"].isin(selected_students)]
                            .groupby("NAME")["DATE"]
                            .nunique()
                            .reset_index(name="Days Present")
                        )

                        if total_days > 0:
                            comp["Attendance %"] = (comp["Days Present"] / total_days * 100).round(2)

                        st.dataframe(comp.set_index("NAME"))

                        st.bar_chart(
                            data=comp.set_index("NAME")["Attendance %"],
                            use_container_width=True
                        )
                    else:
                        st.info("Select at least one student to compare.")
                else:
                    st.info("Attendance data is missing NAME/DATE columns.")
            else:
                st.info("No attendance data found yet for comparison.")

        # ---------- DANGER ZONE: ERASE ALL DATA ----------
        st.markdown("---")
        st.markdown("### 🧨 Danger Zone: Erase All Data")
        st.warning(
            "This will permanently delete **all registered faces** and **all attendance records** "
            "stored so far. This action cannot be undone."
        )

        if st.button("⚠️ Erase ALL data (faces + attendance)", key="erase_all_data_main"):
            st.session_state.erase_all_confirm = True

        if st.session_state.erase_all_confirm:
            st.error("Are you absolutely sure you want to ERASE ALL DATA? This cannot be undone.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, erase everything", key="erase_all_data_confirm"):
                    try:
                        data_dir = Path("Data")
                        for fname in ["names.pkl", "faces_data.pkl"]:
                            fpath = data_dir / fname
                            if fpath.exists():
                                fpath.unlink()

                        attendance_dir = Path(ATTENDANCE_DIR)
                        if attendance_dir.exists():
                            for f in attendance_dir.glob("Attendance_*.csv"):
                                f.unlink()
                            try:
                                if not any(attendance_dir.iterdir()):
                                    attendance_dir.rmdir()
                            except Exception:
                                pass

                        st.success("✅ All face data and attendance records have been erased.")
                        st.info("Please refresh or restart the app before using it again.")
                    except Exception as e:
                        st.error(f"Error while erasing data: {e}")
                    finally:
                        st.session_state.erase_all_confirm = False

            with col2:
                if st.button("❌ Cancel", key="erase_all_data_cancel"):
                    st.session_state.erase_all_confirm = False
                    st.info("Erase operation cancelled.")
