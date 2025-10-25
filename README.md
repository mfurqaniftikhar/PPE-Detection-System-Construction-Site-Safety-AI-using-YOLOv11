# 🦺 PPE Detection System using YOLOv11 + FastAPI

An **AI-based safety monitoring system** that uses **YOLOv11** to detect **Personal Protective Equipment (PPE)** such as **helmets, vests, and masks** from video streams or CCTV footage.  

The project ensures **construction site safety** by identifying workers who are missing protective gear — marking them with a **red bounding box** and triggering an **alarm**.

---

## 🚧 Key Features
✅ Detects PPE: Helmet, Vest, Mask  
✅ Works with live or recorded video  
✅ Red box & alarm for missing PPE  
✅ Built with **YOLOv11** for high accuracy  
✅ Backend powered by **FastAPI**  
✅ Frontend supports HTML/Streamlit UI  

---

## 🧠 Tech Stack
- **YOLOv11** (Object Detection)
- **FastAPI** (Backend)
- **OpenCV** (Video Processing)
- **PyTorch** (Model Framework)
- **HTML / Streamlit** (Frontend)
- **Playsound / Sounddevice** (Alarm trigger)
- **Python 3.10+**

---

## 🗂️ Project Structure
PPE-Detection-YOLOv11/
│
├── detection/ # Model & detection logic
│ ├── detection.py # YOLOv11 inference script
│ ├── best.pt # Trained model weights
│ └── pycache/ # Cache
│
├── full_backend/ # FastAPI backend
│ ├── main.py # FastAPI app file
│ ├── run.py # Run backend server
│
├── frontend/ # Frontend UI
│ └── index.html # Web interface
│
├── alarm.mp3 # Alarm sound for PPE violation
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mfurqaniftikhar/PPE-Detection-YOLOv11.git
cd PPE-Detection-YOLOv11
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
3️⃣ Install Required Packages
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Backend (FastAPI)
Navigate to the backend folder:

bash
Copy code
cd full_backend
uvicorn main:app --reload
The FastAPI server will start at:

cpp
Copy code
http://127.0.0.1:8000
5️⃣ Open the Frontend
Open frontend/index.html in your browser
or
connect it with your Streamlit app (frontend/app.py) if available.

🧩 How It Works
The video feed (live camera or file) is sent to the FastAPI backend.

YOLOv11 (inside detection/detection.py) processes each frame to detect:

Helmet 🪖

Vest 👕

Mask 😷

Bounding boxes are drawn:

🟩 Green Box → PPE detected (Safe)

🟥 Red Box → Missing PPE (Unsafe → Alarm triggered)

The processed frame is displayed in the frontend.

🧠 Model Info
Model: YOLOv11

Framework: PyTorch

Dataset: Custom PPE dataset (construction workers)

Classes: ['Helmet', 'Vest', 'Mask', 'Person']

Confidence threshold: 0.5

FPS: ~30 (GPU-based)

📸 Example Output
Input	Detection

🔔 Alarm System
Whenever the system detects a worker without Helmet, Vest, or Mask:

Draws a red bounding box

Plays the alarm.mp3 sound file
This ensures instant alert for on-site supervisors.

🧱 API Endpoints (FastAPI)
Method	Endpoint	Description
POST	/detect	Upload video/image for PPE detection
GET	/health	Health check endpoint


📈 Performance
PPE Item	Precision	Recall	F1-Score
Helmet	0.98	0.97	0.975
Vest	0.96	0.95	0.955
Mask	0.94	0.93	0.935

👨‍💻 Author
Furqan Iftikhar
AI Engineer | Deep Learning Enthusiast
📧 mfurqaniftikhar00@gmail.com


🪪 License
This project is open-source under the MIT License.

🌟 Future Improvements
📱 Mobile app integration

☁️ Cloud-based live monitoring dashboard

📊 PPE compliance reporting system

🚨 SMS/Email alert for non-compliance

❤️ Acknowledgements
Ultralytics YOLOv11
FastAPI
OpenCV
All open-source contributors in AI safety systems.
yaml

