import os
import math
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


tf.keras.mixed_precision.set_global_policy('float32')

processed_dir = "processed_dataset"
GRAPH_DIR = "graphs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(GRAPH_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f"TensorFlow Version: {tf.__version__}")


print("\nLoading datasets...")

if not os.path.exists(f"{processed_dir}/train"):
    print(f"Error: '{processed_dir}' not found. Run 'train1.py' once to generate it.")
    exit()

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{processed_dir}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = val_ds.class_names
print(f"Classes: {class_names}")

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("Extracting validation labels (this takes a moment)...")
y_true = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)


model_files = []
search_dirs = ['.', 'models']

for d in search_dirs:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith(('.h5', '.keras')):
                full_path = os.path.join(d, f)
                model_files.append(full_path)

if not model_files:
    print("\nERROR: No .h5 or .keras files found!")
    print("   Please put your model files in this folder or a 'models' subfolder.")
    exit()

results = []

print(f"\nScanning {len(model_files)} models...\n")
print(f"{'MODEL NAME':<30} | {'VAL ACC':<8} | {'STATUS'}")
print("-" * 65)

for filepath in model_files:
    filename = os.path.basename(filepath)
    
    try:
        model = load_model(filepath, compile=False)
        model.compile(metrics=['accuracy'])
        
        _, train_acc = model.evaluate(train_ds, verbose=0)
        _, val_acc = model.evaluate(val_ds, verbose=0)
        
        y_pred_probs = model.predict(val_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        
        train_pct = train_acc * 100
        val_pct = val_acc * 100
        gap = train_pct - val_pct
        
        status = "OK"
        if val_pct < 50: status = "Underfit" 
        if gap > 10: status = "Overfit"     

        print(f"{filename:<30} | {val_pct:.1f}%    | {status}")
        
        results.append({
            'name': filename,
            'train_acc': train_pct,
            'val_acc': val_pct,
            'gap': gap,
            'cm': cm
        })
        
        tf.keras.backend.clear_session()
        del model

    except Exception as e:
        print(f"{filename:<30} | {'---':<8} | Skipped (Incompatible)")

if not results:
    print("\nNo successful models found. Exiting.")
    exit()

results.sort(key=lambda x: x['val_acc'], reverse=True)
df = pd.DataFrame(results)


num_models = len(results)
cols = 3  
rows = math.ceil(num_models / cols)

fig_height = 7 + (rows * 4) 
fig = plt.figure(figsize=(18, fig_height))
fig.suptitle(f"Model Benchmark Report ({timestamp})", fontsize=20, fontweight='bold')

ax1 = plt.subplot2grid((rows + 2, cols), (0, 0), colspan=1, rowspan=2)
ax2 = plt.subplot2grid((rows + 2, cols), (0, 1), colspan=2, rowspan=2) 

ax1.barh(df['name'], df['val_acc'], color='teal')
ax1.set_xlabel('Validation Accuracy (%)')
ax1.set_title('Leaderboard (Best Performance)')
ax1.set_xlim(0, 105)
ax1.invert_yaxis()
ax1.grid(axis='x', linestyle='--', alpha=0.5)
for i, val in enumerate(df['val_acc']):
    ax1.text(val + 1, i, f"{val:.1f}%", va='center', fontweight='bold')

y_pos = np.arange(len(df))
h = 0.35
ax2.barh(y_pos - h/2, df['train_acc'], h, label='Training (Memory)', color='lightgray')
ax2.barh(y_pos + h/2, df['val_acc'], h, label='Validation (Exam)', color='dodgerblue')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(df['name'])
ax2.set_title('Overfitting Diagnosis (Gap Test)')
ax2.legend()
ax2.grid(axis='x', linestyle='--', alpha=0.5)
ax2.set_xlim(0, 110)
ax2.invert_yaxis()

for i, (train, val) in enumerate(zip(df['train_acc'], df['val_acc'])):
    gap = train - val
    color = 'red' if gap > 10 else ('green' if gap < 5 else 'orange')
    x_pos = max(train, val) + 2
    ax2.text(x_pos, i, f"Gap: {gap:.1f}%", va='center', color=color, fontweight='bold')

for i, res in enumerate(results):
    row_idx = 2 + (i // cols)
    col_idx = i % cols
    
    ax = plt.subplot2grid((rows + 2, cols), (row_idx, col_idx))
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    
    ax.set_title(f"{res['name']}\n(Acc: {res['val_acc']:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

save_path = f"{GRAPH_DIR}/Report_{timestamp}.png"
plt.savefig(save_path)
print(f"\nDashboard saved: {save_path}")
plt.show()