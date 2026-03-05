import os
from PIL import Image
import shutil

# ==============================
# CONFIGURATION
# ==============================
INPUT_DIR = "raw_data"          # Your huge 5.9GB folder
OUTPUT_DIR = "submission_data"  # New small folder for zip
MAX_SIZE = 600                  # Resize images to max 600px (Models only need 224)
QUALITY = 75                    # JPG Quality (75 is standard)

# ==============================
# COMPRESSION LOOP
# ==============================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"Compressing '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
print("This may take a minute...\n")

total_saved = 0
count = 0

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            
            # 1. Setup Paths
            # Get the class folder name (e.g., 'banana')
            relative_path = os.path.relpath(root, INPUT_DIR)
            target_folder = os.path.join(OUTPUT_DIR, relative_path)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            input_path = os.path.join(root, file)
            output_path = os.path.join(target_folder, os.path.splitext(file)[0] + ".jpg")
            
            try:
                # 2. Open & Resize
                with Image.open(input_path) as img:
                    # Convert to RGB (fixes PNG transparency issues)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Resize while keeping aspect ratio
                    img.thumbnail((MAX_SIZE, MAX_SIZE))
                    
                    # 3. Save as Optimized JPG
                    img.save(output_path, "JPEG", quality=QUALITY, optimize=True)
                    
                    # Calculate savings
                    original_size = os.path.getsize(input_path)
                    new_size = os.path.getsize(output_path)
                    total_saved += (original_size - new_size)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Processed {count} images...")

            except Exception as e:
                print(f"⚠️ Skipped {file}: {e}")

# Convert bytes to GB
saved_gb = total_saved / (1024 * 1024 * 1024)
print(f"\n✅ DONE! Processed {count} images.")
print(f"📉 You saved {saved_gb:.2f} GB of space!")
print(f"👉 Now Zip the '{OUTPUT_DIR}' folder.")