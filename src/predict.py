# src/predict.py
import cv2
import numpy as np
import pandas as pd

def filter_probable_microplastics(df, img,
                                  conf_thresh=0.65,
                                  area_thresh=30,
                                  circ_thresh=0.5,
                                  solidity_thresh=0.75,
                                  extent_thresh=0.4,
                                  aspect_ratio_max=4.0,
                                  box_color=(255, 165, 0)):
    """
    Filters detections to keep only those highly likely to be microplastics 
    based on custom shape, bounding metrics, and stain confidence scores.
    """
    probable_img = img.copy()

    # If the initial dataframe is empty, short-circuit immediately
    if df.empty:
        return probable_img, df

    # Apply tight morphological multi-conditional filter mask
    mask = (df["confidence"] >= conf_thresh) & \
           (df["area_px"] >= area_thresh) & \
           (df["circularity"] >= circ_thresh) & \
           (df["solidity"] >= solidity_thresh) & \
           (df["extent"] >= extent_thresh) & \
           (df["aspect_ratio"] <= aspect_ratio_max)

    probable_df = df[mask].reset_index(drop=True)

    # If no particles remain after filtering, return the clean image and empty df
    if len(probable_df) == 0:
        return probable_img, probable_df

    # Draw thicker, high-contrast bounding boxes for valid particles
    for _, row in probable_df.iterrows():
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        cv2.rectangle(probable_img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(probable_img, f"{int(row['id'])}:{row['confidence']:.2f}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return probable_img, probable_df
