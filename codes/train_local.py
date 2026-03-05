import os
import shutil
import splitfolders
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS! Found GPU: {gpus[0].name}")
else:
    print("⚠️ WARNING: No GPU found. Training might be slow.")

# 1. AUTO-ORGANIZE & CLEANUP
raw_dir = "raw_data"
processed_dir = "processed_dataset"
target_classes = ["banana", "dragonfruit", "unknown"] 

if os.path.exists(processed_dir):
    print(f"Removing old '{processed_dir}' to ensure fresh split...")
    shutil.rmtree(processed_dir)

if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
    print(f"❌ ERROR: '{raw_dir}' is missing or empty!")
    print(f"Please create '{raw_dir}' and put folders {target_classes} inside.")
    exit()

# 2. SPLIT DATA 
print("\nSplitting images into Train (80%), Val (10%), Test (10%)...")

splitfolders.ratio(raw_dir, output=processed_dir, 
                   seed=1337, ratio=(.8, .1, .1), group_prefix=None)

print("✅ Data split successfully!")

# 3. LOAD DATASET
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 4.AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.Lambda(lambda x: tf.image.random_saturation(x, 0.7, 1.3))
])

# 5. CNN MODEL
learning_rate = 1e-4
l2_strength = 1e-4    

print("\nBuilding Model...")

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    # Block 4 
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    # Dense Layers
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dropout(0.5), 
    
    layers.Dense(len(class_names))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

# 6. TRAIN
print("\nStarting Training...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,          
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=4, 
    min_lr=1e-6,
    verbose=1
)

epochs = 50 

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# 7.RESULTS
model.save('Jan_23_model2.h5') 
print("\n✅ Model saved as 'Jan21_balance_model.h5'")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo-', label='Training acc')
plt.plot(epochs_range, val_acc, 'r-', label='Validation acc')
plt.title('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo-', label='Training loss')
plt.plot(epochs_range, val_loss, 'r-', label='Validation loss')
plt.title('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.savefig('training_graph2.png')
print("✅ Graphs saved as 'training_graph.png'")
plt.show()