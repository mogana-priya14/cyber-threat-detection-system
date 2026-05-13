import tensorflow as tf
import numpy as np
import joblib
from tkinter import messagebox

model = tf.keras.models.load_model("cnn_lstm_network_attack_model.h5")
scaler = joblib.load("scaler.save")
encoders = joblib.load("label_encoders.save")

name ="UDP,1754,3.09,443,443,1626,1703,FIN,19.2,1913.5,256,37,44,128,256,985,1471,0"
values = name.split(',')

values[0] = encoders['Protocol'].transform([values[0]])[0]
values[7] = encoders['Flags'].transform([values[7]])[0]

float_array = np.array(values, dtype=float)
reshaped_array = float_array.reshape(1, -1)

scaled_input = scaler.transform(reshaped_array)

TIME_STEPS = 5

sequence_input = np.repeat(
    scaled_input[np.newaxis, :, :],
    TIME_STEPS,
    axis=1
)

pred = model.predict(sequence_input)
score = pred[0][0]

print("Prediction score:", score)

if score > 0.5:
    Answer = "🚨 Attack"
else:
    Answer = "✅ Normal"

#messagebox.showinfo("Prediction Result", Answer)

print(Answer)
