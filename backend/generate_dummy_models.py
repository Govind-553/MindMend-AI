import h5py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# === Dummy Classes for Pickling ===
class DummyKeystrokeModel:
    def predict(self, X):
        return ["focused"] * len(X)

class DummySpeechEmotionModel:
    def predict(self, X):
        return ["happy"] * len(X)

# === 1. Dummy FER2013 model (.h5) ===
with h5py.File("models/fer2013_model.h5", "w") as f:
    f.create_dataset("weights", data=np.random.rand(64, 64))
    f.attrs["note"] = "Dummy FER2013 model for hackathon demo"

# === 2. Dummy Keystroke Model (.pkl) ===
with open("models/keystroke_model.pkl", "wb") as f:
    pickle.dump(DummyKeystrokeModel(), f)

# === 3. Dummy Speech Emotion Model (.pkl) ===
with open("models/speech_emotion_model.pkl", "wb") as f:
    pickle.dump(DummySpeechEmotionModel(), f)

# === 4. Dummy Speech Scaler (.pkl) ===
scaler = StandardScaler()
scaler.fit(np.random.rand(10, 5))  # Fit on random data
with open("models/speech_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ All dummy models generated successfully!")
