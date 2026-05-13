# Importing essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
# Loading the dataset
df = pd.read_csv('../Dataset/CyberThreat.csv')


print(df)



#import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'your_data' is your DataFrame and 'your_column' is the column you want to plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create a count plot
ax = sns.countplot(x='Label', data=df)

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()



df=df.drop(columns='Timestamp')
df=df.drop(columns='Source_IP')
df=df.drop(columns='Destination_IP')
df=df.drop(columns='Protocol')
df=df.drop(columns='Attack_Type')
df=df.drop(columns='Flags')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = clean_dataset(df)
from sklearn.model_selection import train_test_split
X = df.drop(columns='Label')
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for CNN (assuming 1D convolution)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN-LSTM model
#https://www.kaggle.com/code/abdallahmohamedamin/sentiment-analysis-using-cnn-lstm-and-cnn-lstm#3--LSTM-CNN-Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50))
model.add(Dense(8, activation='softmax'))  # Adjust the output neurons based on the classification task
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history2 = model.fit(X_train_cnn, y_train, epochs=100, batch_size=64, validation_data=(X_test_cnn, y_test))

model.save("model.h5", include_optimizer=False)
# Evaluate the model

acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Model Loss
loss_train = history2.history['loss']
loss_val = history2.history['val_loss']
plt.plot(epochs, loss_train, label='Training Loss')
plt.plot(epochs, loss_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



loss, accuracy = model.evaluate(X_test_cnn, y_test)
print(f'Test accuracy: {accuracy}')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


Y_pred = model.predict(X_test_cnn)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)





