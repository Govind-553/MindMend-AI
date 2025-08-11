# 🧠 MindMend AI

AI-powered Wellness Monitoring System for real-time mental well-being analysis using facial expressions, keystroke dynamics, and speech emotion cues.

*Currently in mock prediction mode for demos—structure supports future integration with TensorFlow, Keras, and Scikit-Learn.*

---

## Project Structure

    MindMend-AI/
    │
    ├── backend/
    │ ├── app.py # Flask backend application
    │ ├── analyzers/ # Facial, Speech, Keystroke analyzers
    │ ├── generate_dummy_models.py # Create mock models
    │ ├── requirements.txt
    │ └── ...
    │
    ├── models/ # Placeholder/dummy model files (.pkl, .h5)
    │
    └── README.md

---

## Setup Instructions

### 1. Clone the repository

    git clone https://github.com/yourusername/MindMend-AI.git
    cd MindMend-AI/backend


### 2. Create & activate a virtual environment

<details>
<summary>Windows (PowerShell)</summary>

    python -m venv venv
    .\venv\Scripts\activate

</details>


### 3. Install dependencies

(Mock demo does not require TensorFlow/ML libraries)

    pip install -r requirements.txt

<sub>For real model integration, install TensorFlow and other required ML libraries.</sub>

### 4. Generate dummy models

   cd backend
   python generate_dummy_models.py

This creates mock model files in the `models/` directory:
- `fer2013_model.h5` – Dummy FER2013 model
- `keystroke_model.pkl` – Mock keystroke dynamics model
- `speech_emotion_model.pkl` – Mock speech emotion model
- `speech_scaler.pkl` – Mock scaler for speech preprocessing

### 5. Run the backend server

    python app.py


    Server runs at: http://127.0.0.1:5000  
You’ll see logs confirming that mock predictions are being used.

---

## Mock Prediction Mode

- Facial, keystroke, and speech analyzers generate random emotions.
- Placeholder `.pkl` / `.h5` model files used.
- Runs quickly—no heavy ML dependencies.

---

## Future Model Replacements

- TensorFlow/Keras CNN for Facial Emotion Recognition
- Scikit-learn models for Keystroke & Speech Analysis
- Proper pipelines for feature extraction, preprocessing, and scaling

---

## Roadmap

- [ ] Replace mock models with trained versions
- [ ] Integrate real FER2013 CNN for facial recognition
- [ ] Fine-tune speech and keystroke models
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Real-time data streaming

---

## Team

| Name              | Role                                 |
|-------------------|--------------------------------------|
| Govind Choudhari  | Team Lead, Full-Stack Developer      |
| Abhiruchi Kunte   | AI/ML Developer, Model Tuning        |
| Sahil Kale        | UI/UX Designer, Frontend Developer   |
| Nishank Jain      | Backend Developer & Integration      |

---

## Disclaimer

*Prototype for hackathon/demo purposes only.  
Not intended for medical diagnosis or clinical use without further research, validation, and testing.*
