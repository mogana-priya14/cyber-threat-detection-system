import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2



df = pd.read_csv("Dataset/CyberThreat.csv")
print("Dataset Shape:", df.shape)



df.drop([
    'Timestamp',
    'Source_IP',
    'Destination_IP',
    'Label'      # remove Label completely
], axis=1, inplace=True)



feature_encoders = {}

for col in ['Protocol', 'Flags']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    feature_encoders[col] = le

joblib.dump(feature_encoders, "label_encoders.save")
print("Feature encoders saved")



attack_encoder = LabelEncoder()
df['Attack_Type'] = attack_encoder.fit_transform(df['Attack_Type'])

joblib.dump(attack_encoder, "attack_type_encoder.save")
print("Attack_Type encoder saved")

num_classes = len(np.unique(df['Attack_Type']))
print("Number of Attack Classes:", num_classes)



X = df.drop('Attack_Type', axis=1)
y = df['Attack_Type'].values



scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.save")
print("Scaler saved")



def create_sequences(X, y, time_steps=15):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 15
X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

print("Sequence Shape:", X_seq.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    shuffle=False
)



weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)



model = Sequential()

model.add(Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu',
    kernel_regularizer=l2(0.001),
    input_shape=(X_train.shape[1], X_train.shape[2])
))

model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(
    64,
    dropout=0.3,
    recurrent_dropout=0.3
))

model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



"""early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)"""

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-5
)



history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    #callbacks=[early_stop, lr_reduce]
)



model.save("cnn_lstm_network_attack_model.h5")
print("Model saved successfully")



loss, acc = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", round(acc * 100, 2), "%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=attack_encoder.classes_
))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=attack_encoder.classes_,
    yticklabels=attack_encoder.classes_
)

plt.title("Multi-Class Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.grid(True)
plt.show()



plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.grid(True)
plt.show()
