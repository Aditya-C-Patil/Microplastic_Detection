# main.py
import cv2
import os
import pandas as pd
from src.preprocess import process_pair
from src.predict import filter_probable_microplastics

def run_pipeline():
    print("🚀 Running Microplastic Detection Pipeline...")
    
    # 1. Paths to your data files
    img_off_path = "data/sample_led_off.jpg"
    img_on_path = "data/sample_led_on.jpg"
    output_dir = "output"
    
    # Create an output directory to save results if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Read the input images locally
    img_off = cv2.imread(img_off_path, cv2.IMREAD_COLOR)
    img_on = cv2.imread(img_on_path, cv2.IMREAD_COLOR)

    if img_off is None or img_on is None:
        print("❌ Error: Missing input images in the 'data/' directory.")
        print("Please place 'sample_led_off.jpg' and 'sample_led_on.jpg' inside the data folder.")
        return

    # 3. Shape alignment safeguards
    if len(img_off.shape) == 2:
        img_off = cv2.cvtColor(img_off, cv2.COLOR_GRAY2BGR)
    if len(img_on.shape) == 2:
        img_on = cv2.cvtColor(img_on, cv2.COLOR_GRAY2BGR)

    if img_off.shape != img_on.shape:
        img_on = cv2.resize(img_on, (img_off.shape[1], img_off.shape[0]))

    # 4. Step 1: Extract all raw candidate features
    annotated_raw, df_raw, mask = process_pair(img_off, img_on)

    # 5. Step 2: Filter high-confidence microplastics using our predict engine
    final_img, final_df = filter_probable_microplastics(df_raw, annotated_raw)

    # 6. Save data and export visual results without freezing execution
    if final_df.empty:
        print("🔍 Execution Complete: 0 microplastic particles passed the filter thresholds.")
    else:
        # Save annotated image result
        output_img_path = os.path.join(output_dir, "detected_microplastics.jpg")
        cv2.imwrite(output_img_path, final_img)
        
        # Save metrics sheet data log to CSV
        csv_path = os.path.join(output_dir, "detection_metrics.csv")
        final_df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Success! Detected {len(final_df)} highly probable microplastic particles.")
        print(f"📁 Output image saved to: {output_img_path}")
        print(f"📁 Metrics datasheet exported to: {csv_path}\n")
        print(final_df[["id", "type", "area_px", "confidence", "solidity"]].to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
