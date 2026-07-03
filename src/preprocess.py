# src/preprocess.py
import cv2
import numpy as np
import pandas as pd
import math

def process_pair(img_off, img_on, hsv_lower=(5, 80, 50), hsv_upper=(25, 255, 255), min_area_px=30):
    """
    Processes a pair of images (LED off vs LED on) to segment and extract 
    geometrical and color features of Sudan III stained microplastics.
    """
    # Step 1: Red channel subtraction
    red_off = img_off[:, :, 2].astype(np.int16)
    red_on = img_on[:, :, 2].astype(np.int16)
    diff = cv2.subtract(red_on, red_off).astype(np.uint8)

    # Step 2: Noise reduction
    blur = cv2.medianBlur(diff, 3)

    # Step 3: HSV mask on ON frame + combine with subtraction mask
    hsv = cv2.cvtColor(img_on, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    combined_mask = cv2.bitwise_and(color_mask, blur)

    # Step 4: Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 5: Remove small blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    final_mask = np.zeros_like(clean_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            final_mask[labels == i] = 255

    # Step 6: Contours + extended feature extraction
    annotated = img_on.copy()
    out_data = []
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for pid, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Features
        circularity = (4 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        aspect_ratio = float(w) / h if h > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        extent = float(area) / (w * h) if (w * h) > 0 else 0

        # Eccentricity (from ellipse fitting)
        eccentricity = 0
        if len(cnt) >= 5:
            try:
                (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
                eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma > 0 else 0
            except cv2.error:
                eccentricity = 0

        # Hu moments (7 invariant features)
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # Mean red intensity (staining strength) with fallback check
        mask_patch = final_mask[y:y+h, x:x+w]
        img_patch = img_on[y:y+h, x:x+w, 2]
        if np.any(mask_patch == 255):
            mean_red = img_patch[mask_patch == 255].mean()
        else:
            mean_red = 0

        # Confidence score
        color_score = mean_red / 255.0
        shape_score = min(1.0, circularity * 1.5)
        confidence = 0.7 * color_score + 0.3 * shape_score

        # Particle type classification
        if aspect_ratio > 3:
            ptype = "fiber"
        elif circularity > 0.7:
            ptype = "bead"
        else:
            ptype = "fragment"

        out_data.append({
            "id": pid, "x": x, "y": y, "w": w, "h": h,
            "area_px": area, "circularity": circularity,
            "aspect_ratio": aspect_ratio, "solidity": solidity,
            "extent": extent, "eccentricity": eccentricity,
            "mean_red": mean_red, "confidence": confidence, "type": ptype,
            "hu1": hu[0], "hu2": hu[1], "hu3": hu[2],
            "hu4": hu[3], "hu5": hu[4], "hu6": hu[5], "hu7": hu[6]
        })

        # Annotate image frame
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(annotated, f"{pid}:{confidence:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Return empty structured DataFrame if no particles found
    if len(out_data) == 0:
        columns = ["id", "x", "y", "w", "h", "area_px", "circularity", "aspect_ratio", 
                   "solidity", "extent", "eccentricity", "mean_red", "confidence", "type"] + [f"hu{i}" for i in range(1, 8)]
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(out_data)

    return annotated, df, final_mask
