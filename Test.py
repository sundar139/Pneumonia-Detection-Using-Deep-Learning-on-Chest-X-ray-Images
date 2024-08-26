import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
else:
    print("GPU is not available. Using CPU.")

data_dir = "Pneumonia Detection/chest_xray/Dataset"
labels = ['NORMAL', 'PNEUMONIA']

normal_dir = os.path.join(data_dir, "NORMAL")
pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")

print("no. PNEUMONIA:", len(os.listdir(pneumonia_dir)))
print("no. NORMAL:", len(os.listdir(normal_dir)))

data = []
target = []

for label in labels:
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        one_image = cv2.imread(img_path)
        if one_image is not None:
            resize_img = cv2.resize(one_image, (128, 128))
            normalize_img = resize_img / 255.0
            data.append(normalize_img)
            target.append(labels.index(label))

data = np.array(data, dtype=np.float32)
target = np.array(target, dtype=np.float32)

print("Data Type:", type(data))
print("Target Type:", type(target))
print("Data Shape:", data.shape)
print("Target Shape:", target.shape)

def channel_attention(x, reduction_ratio=8):
    channel = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Dense(channel // reduction_ratio, activation='relu')(avg_pool)
    avg_pool = layers.Dense(channel, activation='sigmoid')(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Dense(channel // reduction_ratio, activation='relu')(max_pool)
    max_pool = layers.Dense(channel, activation='sigmoid')(max_pool)
    
    attention = avg_pool + max_pool
    attention = layers.Reshape((1, 1, channel))(attention)
    
    return x * attention

def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return x * attention

def cbam_block(x, reduction_ratio=8):
    x = channel_attention(x, reduction_ratio)
    x = spatial_attention(x)
    return x

input_shape = (128, 128, 3)
inputs = layers.Input(shape=input_shape)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = cbam_block(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = cbam_block(x)

x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = cbam_block(x)

x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = cbam_block(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

x_train, x_temp, y_train, y_temp = train_test_split(data, target, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history_callback = tf.keras.callbacks.History()

model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_val, y_val), 
          callbacks=[history_callback, lr_scheduler])

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.65).astype(int).flatten()

print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=labels))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_callback.history['accuracy'])
plt.plot(history_callback.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history_callback.history['loss'])
plt.plot(history_callback.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
