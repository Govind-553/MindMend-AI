from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---- FER2013 Model ----
model = Sequential([
    Flatten(input_shape=(48, 48, 1)),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy')
model.save("fer2013_model.h5")

# ---- Keystroke Model ----
X_k = np.random.rand(20, 10)
y_k = np.random.choice(['Focused', 'Stressed'], 20)
keystroke_clf = RandomForestClassifier()
keystroke_clf.fit(X_k, y_k)
joblib.dump(keystroke_clf, "keystroke_mdoel.pkl")

# ---- Speech Scaler ----
X_s = np.random.rand(20, 40)
scaler = StandardScaler()
scaler.fit(X_s)
joblib.dump(scaler, "speech_scaler.pkl")

# ---- Speech Emotion Model ----
y_s = np.random.choice(['Happy', 'Sad', 'Angry', 'Neutral'], 20)
speech_clf = RandomForestClassifier()
speech_clf.fit(X_s, y_s)
joblib.dump(speech_clf, "speech_emotion_model.pkl")

print("✅ All dummy models generated successfully!")
