# 📋 Smart Attendance System

A **Face Recognition-based Smart Attendance System** built with **Streamlit**, **OpenCV**, **streamlit-webrtc**, **scikit-learn**, and **MongoDB Atlas**.

This application allows you to:  
- 👤 **Register faces** using your webcam (captures multiple images per person).  
- 🎥 **Take attendance** using real-time face recognition.  
- ☁️ **Store attendance securely** in MongoDB Cloud.  
- 📅 **View attendance for any selected date**.  
- 🔐 **Use Admin Panel** to edit, compare, and analyze attendance.  

---

## 🚀 Features
- Real-time video processing with **streamlit-webrtc**  
- Face detection using **Haar Cascade Classifier**  
- Face recognition using **KNN classifier**  
- Attendance saved in **MongoDB Atlas**  
- Simple, interactive **Streamlit UI**

### 🔥 Enhanced Features
- 📅 View **any day's attendance**  
- ✏️ Edit attendance (Admin panel)  
- 📊 Student analytics  
- ⚖️ Compare students  
- 🔑 Change admin password  
- 🧨 Delete all face + attendance data  

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/OmSingh29/smart-attendance-system
```
```bash
cd smart-attendance-system
```

### 2. Create Virtual Environment (NO conda)
```bash
python -m venv smart_attendance
```
```bash
smart_attendance\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run the App
```bash
streamlit run app.py
```

### 2. Register New Face
- Enter your **name**  
- Click **Start Registration**  
- Camera turns on  
- System captures **5 images**  

### 3. Take Attendance
- Start camera under **Take Attendance**  
- App recognizes registered faces  
- Attendance saved in MongoDB  

### 4. View Attendance
Select any date to view the attendance list.

### 5. Admin Panel
- Edit attendance  
- Student analytics  
- Compare students  
- Change admin password  
- Delete all data  

---

## 📦 Data Storage

### Local Files
- faces_data.pkl  
- names.pkl  
- haarcascade_frontalface_default.xml  

### MongoDB Document Format
```json
{
  "name": "Arin",
  "date": "25-11-2025",
  "time": "09:12:54",
  "timestamp": "2025-11-25T09:12:54+05:30",
  "source": "streamlit_app"
}
```

---

## 🌐 Deployment (Streamlit Cloud)

### TURN/STUN Configuration
```python
rtc_configuration={
  "iceServers":[
    {"urls":["stun:stun.l.google.com:19302"]},
    {"urls":["turn:relay.metered.ca:80"], "username":"open", "credential":"open"}
  ]
}
```

### Streamlit Secrets
```toml
MONGO_URI = "your_mongo_uri"
MONGO_DB_NAME = "smart_attendance"
MONGO_COLLECTION = "attendance_records"
```

---
