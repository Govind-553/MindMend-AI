# MindMend AI 🧠💡

MindMend AI is an AI-powered **Wellness Monitoring System** that analyzes **facial expressions**, **keystroke patterns**, and **speech emotion cues** to assess mental well-being in real-time.  
Currently, the system uses **mock prediction models** for demonstration purposes, but the structure is ready for full integration with **TensorFlow** and other ML frameworks in later stages.

---

## 🚀 Features
- Facial emotion detection (FER2013 architecture – placeholder for now).
- Keystroke dynamics analysis for mood inference.
- Speech emotion recognition.
- Flask-based backend for API services.
- Modular architecture for easy model replacement.
- Mock predictions for hackathon/demo purposes (no heavy ML dependencies required).

---

## 📦 Project Structure
MindMend-AI/
│
├── backend/
│ ├── app.py # Main Flask application
│ ├── analyzers/ # Facial, Speech, Keystroke analyzers
│ ├── generate_dummy_models.py # Script to create mock models
│ ├── requirements.txt
│ └── ...
│
├── models/ # Saved model files (.pkl, .h5) – currently dummy
│
└── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/MindMend-AI.git
cd MindMend-AI/backend
2️⃣ Create & activate a virtual environment
bash
Copy
Edit
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
Since we are using mock models, you do not need TensorFlow or heavy ML libraries for now.
However, the backend still uses core dependencies like Flask, NumPy, etc.

bash
Copy
Edit
pip install -r requirements.txt
(If you later replace with real models, you'll need to install TensorFlow)

4️⃣ Generate dummy models
Run this to create placeholder model files in the models directory:

bash
Copy
Edit
cd backend
python generate_dummy_models.py
This will generate:

fer2013_model.h5 (dummy file)

keystroke_model.pkl (mock keystroke data)

speech_emotion_model.pkl (mock speech data)

speech_scaler.pkl (mock scaler data)

5️⃣ Run the backend server
bash
Copy
Edit
python app.py
The server will start at:

cpp
Copy
Edit
http://127.0.0.1:5000
You should see log messages indicating mock predictions are being used.

🛠 Mock Prediction Mode
Right now, MindMend AI does not load real machine learning models — it uses:

Random emotion generation for facial, keystroke, and speech analysis.

Lightweight .pkl and .h5 placeholders to mimic real models.

This allows fast demo without installing heavy ML dependencies.

Later, you can replace these mock files with:

Trained TensorFlow/Keras models for facial emotion recognition.

Scikit-learn models for keystroke & speech analysis.

Proper pre-processing & scaling pipelines.

📌 Roadmap (Future Scope)
Replace mock models with trained versions.

Integrate real FER2013 CNN model for facial recognition.

Fine-tune speech and keystroke models for accuracy.

Deploy to cloud (AWS/GCP).

👥 Team
Name	Role
Govind Choudhari	Team Lead, Full-Stack Developer
Abhiruchi Kunte	AI/ML Developer, Model Tuning
Sahil Kale	UI/UX Designer, Frontend Developer
Nishank Jain	Backend Developer & Integration

⚠️ Disclaimer
This project is currently a prototype for hackathon/demo purposes.
It is not intended for medical diagnosis or real-time mental health monitoring without further research, testing, and validation.