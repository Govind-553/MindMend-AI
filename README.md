🧠 MindMend AI
AI-Powered Wellness Monitoring System for real-time assessment of mental well-being using facial expressions, keystroke patterns, and speech emotion cues.

Currently, MindMend AI runs in mock prediction mode for demonstration purposes — ready for future integration with TensorFlow, Keras, and Scikit-Learn trained models.

🚀 Features
Facial Emotion Detection (FER2013 CNN architecture – placeholder for now)

Keystroke Dynamics Analysis for mood inference

Speech Emotion Recognition using audio cues

Flask-based API Backend for easy integration

Modular Architecture for swapping mock and real models

Lightweight Demo Mode — no heavy ML dependencies required

📂 Project Structure
text
MindMend-AI/
│
├── backend/
│   ├── app.py                  # Flask backend application
│   ├── analyzers/              # Facial, Speech, Keystroke analyzers
│   ├── generate_dummy_models.py # Script to create mock models
│   ├── requirements.txt
│   └── ...
│
├── models/                     # Placeholder/dummy model files (.pkl, .h5)
│
└── README.md
⚙️ Setup Instructions
1️⃣ Clone the repository
bash
git clone https://github.com/yourusername/MindMend-AI.git
cd MindMend-AI/backend
2️⃣ Create & activate a virtual environment
Windows (PowerShell)
bash
python -m venv venv
.\venv\Scripts\activate
macOS/Linux
bash
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
(Mock mode doesn’t require TensorFlow or heavy ML libraries yet)

bash
pip install -r requirements.txt
When integrating real models, you’ll need to install TensorFlow and other ML libraries.

4️⃣ Generate dummy models
bash
cd backend
python generate_dummy_models.py
This will create mock model files in the models directory:

fer2013_model.h5 – Dummy FER2013 model

keystroke_model.pkl – Mock keystroke dynamics model

speech_emotion_model.pkl – Mock speech emotion model

speech_scaler.pkl – Mock scaler for speech preprocessing

5️⃣ Run the backend server
bash
python app.py
Server runs at:

text
http://127.0.0.1:5000
You’ll see logs confirming that mock predictions are being used.

🛠 Mock Prediction Mode
In current demo mode:

Facial, keystroke, and speech analysis generate random emotions

.pkl and .h5 files act as placeholders

Fast execution without large ML dependencies

🔜 Future Model Replacements
Real TensorFlow/Keras CNN for Facial Emotion Recognition

Scikit-learn models for Keystroke & Speech Analysis

Proper feature extraction, preprocessing, and scaling pipelines

📌 Roadmap
 Replace mock models with trained versions

 Integrate real FER2013 CNN model for facial recognition

 Fine-tune speech and keystroke models for improved accuracy

 Deploy to cloud platforms (AWS, GCP)

 Implement real-time data streaming for continuous monitoring

👥 Team
Name	Role
Govind Choudhari	Team Lead, Full-Stack Developer
Abhiruchi Kunte	AI/ML Developer, Model Tuning
Sahil Kale	UI/UX Designer, Frontend Developer
Nishank Jain	Backend Developer & Integration
⚠️ Disclaimer
This project is a prototype built for hackathon/demo purposes.
It is not intended for medical diagnosis or use in real-time mental health monitoring without further research, validation, and clinical testing.