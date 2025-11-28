# app.py

import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time
import numpy as np
import threading
import av
import warnings
from zoneinfo import ZoneInfo
from pathlib import Path
from face_registration import save_face_data
from take_attendance import load_model, mark_attendance
from db import get_attendance_collection   # <-- NEW: for reading Mongo in app

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("Smart Attendance System")

DEFAULT_ADMIN_PASSWORD = "admin123"  # this one is shown on the site

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
if "last_marked_name" not in st.session_state:
    st.session_state.last_marked_name = ""
if "last_mark_time" not in st.session_state:
    st.session_state.last_mark_time = 0
if "admin_password_current" not in st.session_state:
    st.session_state.admin_password_current = DEFAULT_ADMIN_PASSWORD


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


# ======================= Section 1: Register New Face =======================
with st.container():
    st.subheader("Register New Face")
    st.session_state.new_name = st.text_input(
        "Enter your name:",
        value=st.session_state.get('new_name', ''),
        key="new_name_input"
    )

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


# ======================= Section 2: Take Attendance =======================
with st.container():
    st.subheader("Take Attendance")

    if knn:
        st.info("Click 'START' below to begin attendance.")

        class AttendanceProcessor(VideoTransformerBase):
            def __init__(self):
                # keep track of whose attendance has been marked in this session
                self.attendance_register = set()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)

                recognized_name = ""

                for (x, y, w, h) in faces:
                    crop_img = img[y:y+h, x:x+w]
                    if crop_img.size > 0:
                        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                        output = knn.predict(resized_img)
                        recognized_name = output[0]

                        # draw box + label so you can SEE who it thinks you are
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            img, recognized_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )

                # mark attendance ONCE per person for this run
                if recognized_name and recognized_name not in self.attendance_register:
                    self.attendance_register.add(recognized_name)
                    message = mark_attendance(recognized_name)
                    print(f"[ATTENDANCE] {message}")

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=AttendanceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )

    else:
        st.info("Please register a face before taking attendance.")


# ======================= Section 3: Today's Attendance (from MongoDB) =======================
with st.container():
    st.subheader("Today's Attendance (Quick View)")

    today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y")
    collection = get_attendance_collection()

    if collection is None:
        st.info("Cloud database is not configured. Please set MONGO_* in secrets.")
    else:
        docs = list(collection.find({"date": today_str}))
        if docs:
            df_today = pd.DataFrame(docs)
            if "_id" in df_today.columns:
                df_today.drop(columns=["_id"], inplace=True)
            df_today.rename(columns={"name": "NAME", "time": "TIME", "date": "DATE"}, inplace=True)

            # Add numbering column starting from 1
            df_today.insert(0, "No.", range(1, len(df_today) + 1))

            st.dataframe(df_today[["No.", "NAME", "TIME"]],hide_index=True)

        else:
            st.warning("No attendance has been recorded for today yet.")

# ======================= Section: Attendance Viewer (Any Date) =======================
with st.container():
    st.subheader("View Attendance for Any Date")

    collection = get_attendance_collection()
    if collection is None:
        st.info("Cloud database is not configured.")
    else:
        # choose a date
        selected_date = st.date_input("Select date", datetime.now(ZoneInfo("Asia/Kolkata")).date())
        selected_date_str = selected_date.strftime("%d-%m-%Y")

        # fetch docs for that date
        docs = list(collection.find({"date": selected_date_str}))

        if docs:
            df_sel = pd.DataFrame(docs)
            if "_id" in df_sel.columns:
                df_sel.drop(columns=["_id"], inplace=True)
            df_sel.rename(columns={"name": "NAME", "time": "TIME", "date": "DATE"}, inplace=True)
            df_sel = df_sel[["NAME", "TIME"]]

            # numbering from 1
            df_sel.insert(0, "No.", range(1, len(df_sel) + 1))

            st.success(f"Attendance for {selected_date_str}")
            st.dataframe(df_sel, hide_index=True)
        else:
            st.warning(f"No attendance found for {selected_date_str}.")

# ======================= Section 4: Admin Panel (MongoDB-based) =======================
with st.container():
    st.subheader("Admin Panel")

    if not st.session_state.is_admin:
        # Show ONLY the default password, never the changed one
        st.caption(f"Default admin password (first time): **{DEFAULT_ADMIN_PASSWORD}**")
        st.caption("If you have changed the password earlier, please use that new password (it is not shown here).")

        password_input = st.text_input(
            "Enter admin password",
            type="password",
            key="admin_password_input"
        )

        if st.button("Login as Admin"):
            if password_input == st.session_state.admin_password_current:
                st.session_state.is_admin = True
                st.success("Admin mode enabled. Loading admin tools...")
                st.rerun()
            else:
                st.error("Incorrect admin password.")
    else:
        st.success("Admin mode enabled")

        # ----- Change admin password (only visible in admin mode) -----
        with st.expander("Change admin password"):
            st.write(
                "The default password is only for first-time use. You can set a new admin password below. "
                "For security reasons, the new password will NOT be shown on the page."
            )

            new_pw = st.text_input(
                "New admin password",
                type="password",
                key="new_admin_pw"
            )
            new_pw_confirm = st.text_input(
                "Confirm new admin password",
                type="password",
                key="new_admin_pw_confirm"
            )

            if st.button("Update admin password"):
                if not new_pw:
                    st.error("New password cannot be empty.")
                elif new_pw != new_pw_confirm:
                    st.error("Passwords do not match.")
                else:
                    st.session_state.admin_password_current = new_pw
                    st.success("Admin password updated successfully. Use the new password next time you log in.")

        if st.button("Exit Admin Mode"):
            st.session_state.is_admin = False
            st.info("You have exited admin mode.")
            st.rerun()

        tab1, tab2, tab3 = st.tabs(["Edit Attendance", "Student Analytics", "Compare Students"])

        # ---------- TAB 1: EDIT ATTENDANCE (Mongo) ----------
        with tab1:
            st.markdown("#### Edit / Add / Delete Attendance Records")

            today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
            edit_date = st.date_input("Select date to edit", value=today_ist, key="admin_edit_date")
            edit_date_str = edit_date.strftime("%d-%m-%Y")

            collection = get_attendance_collection()
            if collection is None:
                st.info("Cloud database is not configured.")
            else:
                docs = list(collection.find({"date": edit_date_str}))
                if docs:
                    df_edit = pd.DataFrame(docs)
                    if "_id" in df_edit.columns:
                        df_edit.drop(columns=["_id"], inplace=True)
                    df_edit.rename(
                        columns={"name": "NAME", "time": "TIME", "date": "DATE"},
                        inplace=True
                    )
                    df_edit = df_edit[["NAME", "TIME"]]
                else:
                    df_edit = pd.DataFrame(columns=["NAME", "TIME"])

                st.info("You can add new rows or modify/delete existing rows below.")
                edited_df = st.data_editor(
                    df_edit,
                    num_rows="dynamic",
                    key="admin_editor"
                )

                if st.button("Save changes for selected date"):
                    try:
                        # Replace all records for that date with edited_df
                        collection.delete_many({"date": edit_date_str})

                        for _, row in edited_df.iterrows():
                            name_val = str(row.get("NAME", "")).strip()
                            time_val = str(row.get("TIME", "")).strip()
                            if not name_val or not time_val:
                                continue

                            # rebuild a basic timestamp (optional)
                            ts = datetime.now(ZoneInfo("Asia/Kolkata"))
                            doc = {
                                "name": name_val,
                                "date": edit_date_str,
                                "time": time_val,
                                "timestamp": ts.isoformat(),
                                "source": "admin_edit"
                            }
                            collection.insert_one(doc)

                        st.success(f"Saved changes for {edit_date_str}")
                        st.caption(
                            "- To add attendance of a person who has not come: add a new row with NAME and TIME.\n"
                            "- To delete a record: remove that row from the table before saving."
                        )

                        # 🔁 reload the page so the table reflects the new data immediately
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error saving changes: {e}")

        # ---------- TAB 2: STUDENT ANALYTICS (Mongo) ----------
        with tab2:
            st.markdown("#### Analytics for a Single Student")

            collection = get_attendance_collection()
            if collection is None:
                st.info("Cloud database is not configured.")
            else:
                docs = list(collection.find({}))
                if not docs:
                    st.info("No attendance data found yet.")
                else:
                    df_all = pd.DataFrame(docs)
                    if "_id" in df_all.columns:
                        df_all.drop(columns=["_id"], inplace=True)
                    df_all.rename(
                        columns={"name": "NAME", "date": "DATE", "time": "TIME"},
                        inplace=True
                    )

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

        # ---------- TAB 3: COMPARE STUDENTS (Mongo) ----------
        with tab3:
            st.markdown("#### Compare Students")

            collection = get_attendance_collection()
            if collection is None:
                st.info("Cloud database is not configured.")
            else:
                docs = list(collection.find({}))
                if not docs:
                    st.info("No attendance data found yet for comparison.")
                else:
                    df_all_cmp = pd.DataFrame(docs)
                    if "_id" in df_all_cmp.columns:
                        df_all_cmp.drop(columns=["_id"], inplace=True)
                    df_all_cmp.rename(
                        columns={"name": "NAME", "date": "DATE", "time": "TIME"},
                        inplace=True
                    )

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

        # ---------- DANGER ZONE: ERASE ALL DATA ----------
        st.markdown("---")
        st.markdown("### Reset: Erase All Data")
        st.warning(
            "This will permanently delete **all registered faces** and **all attendance records** "
            "stored so far (both MongoDB + local face data). This action cannot be undone."
        )

        if st.button(" Erase ALL data (faces + attendance)", key="erase_all_data_main"):
            st.session_state.erase_all_confirm = True

        if st.session_state.erase_all_confirm:
            st.error("Are you absolutely sure you want to ERASE ALL DATA? This cannot be undone.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, erase everything", key="erase_all_data_confirm"):
                    try:
                        # Delete face data pickle files
                        data_dir = Path("Data")
                        for fname in ["names.pkl", "faces_data.pkl"]:
                            fpath = data_dir / fname
                            if fpath.exists():
                                fpath.unlink()

                        # Delete all attendance docs from Mongo
                        collection = get_attendance_collection()
                        if collection is not None:
                            collection.delete_many({})
                        st.success("All face data and attendance records have been erased.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while erasing data: {e}")
                    finally:
                        st.session_state.erase_all_confirm = False

            with col2:
                if st.button("❌ Cancel", key="erase_all_data_cancel"):
                    st.session_state.erase_all_confirm = False
                    st.info("Erase operation cancelled.")
