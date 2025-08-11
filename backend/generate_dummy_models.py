import pickle
import os
import h5py
import numpy as np

models_dir = "../../models"
os.makedirs(models_dir, exist_ok=True)

# Dummy facial model (.h5 just to exist, no TF needed)
with h5py.File(os.path.join(models_dir, "fer2013_model.h5"), 'w') as f:
    f.create_dataset("dummy_data", data=np.zeros((1, 1)))

# Dummy keystroke model (dict, not class)
with open(os.path.join(models_dir, "keystroke_model.pkl"), "wb") as f:
    pickle.dump({"type": "dummy_keystroke"}, f)

# Dummy speech model (dict, not class)
with open(os.path.join(models_dir, "speech_emotion_model.pkl"), "wb") as f:
    pickle.dump({"type": "dummy_speech"}, f)

# Dummy speech scaler (dict, not real scaler)
with open(os.path.join(models_dir, "speech_scaler.pkl"), "wb") as f:
    pickle.dump({"type": "dummy_scaler"}, f)

print("Dummy models generated successfully.")
