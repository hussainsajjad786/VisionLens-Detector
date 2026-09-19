"""Lightweight image gate for anterior-segment eye photographs.

This rejects ordinary photos before they reach the cataract classifier. It looks
for a large, near-central circular iris/pupil region; it is not a diagnosis.
"""
import io
import cv2
import numpy as np
from PIL import Image


def validate_eye_image(raw: bytes) -> None:
    try:
        rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as error:
        raise ValueError("The uploaded file is not a readable image.") from error

    height, width = rgb.shape[:2]
    if min(height, width) < 160:
        raise ValueError("Image is too small. Upload a clear close-up eye image at least 160 pixels wide.")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    smallest_side = min(height, width)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=smallest_side * 0.28,
        param1=90, param2=35, minRadius=max(15, int(smallest_side * 0.10)),
        maxRadius=int(smallest_side * 0.49),
    )
    if circles is None:
        raise ValueError("No eye/iris region was detected. Upload a clear, close-up photograph of one eye only.")

    # An iris in a screening image should appear near the centre, not as a tiny
    # circular object at an edge (for example, a wheel, logo, or coin).
    for x, y, radius in np.round(circles[0]).astype(int):
        central = width * 0.15 <= x <= width * 0.85 and height * 0.15 <= y <= height * 0.85
        proportionate = radius >= smallest_side * 0.10
        if central and proportionate:
            return
    raise ValueError("This does not appear to be a close-up eye image. Please upload a clear eye photograph.")
