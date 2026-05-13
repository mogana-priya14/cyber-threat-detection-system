import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import socket
import struct

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv("Dataset/CyberThreat.csv")
print("Dataset Shape:", df.shape)


# ======================================================
# FEATURE ENGINEERING
# ======================================================

# ---- Convert Timestamp to useful features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday

df.drop('Timestamp', axis=1, inplace=True)


# ---- Convert IP addresses to numeric
def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except:
        return 0

df['Source_IP'] = df['Source_IP'].apply(ip_to_int)
df['Destination_IP'] = df['Destination_IP'].apply(ip_to_int)


# ======================================================
# DROP UNUSED COLUMN
# ======================================================
df.drop('Label', axis=1, inplace=True)


# ======================================================
# ENCODE CATEGORICAL FEATURES
# ======================================================
for col in ['Protocol', 'Flags']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# ======================================================
# ENCODE TARGET
# ======================================================
attack_encoder = LabelEncoder()
df['Attack_Type'] = attack_encoder.fit_transform(df['Attack_Type'])

print("Classes:", attack_encoder.classes_)


# ======================================================
# SPLIT FEATURES AND TARGET
# ======================================================
X = df.drop('Attack_Type', axis=1)
y = df['Attack_Type']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================================
# TRAIN RANDOM FOREST
# ======================================================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ======================================================
# EVALUATION
# ======================================================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=attack_encoder.classes_
))


# ======================================================
# CONFUSION MATRIX
# ======================================================
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

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ======================================================
# FEATURE IMPORTANCE
# ======================================================
importances = model.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# ======================================================
# SAVE MODEL
# ======================================================
joblib.dump(model, "cyber_random_forest_model.pkl")
joblib.dump(attack_encoder, "attack_type_encoder.save")

print("Model saved successfully")
