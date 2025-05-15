import numpy as np
import os
import cv2
from pyts.image import RecurrencePlot

# --- Load Data ---
image_dir = 'image'
os.makedirs(image_dir, exist_ok=True)

data = np.load('generated_dataset/corr_dataset/corr_15dB.npy')
pre_idx = 1000
x_corr = data[pre_idx, :1024]
x_recurrence_plot = x_corr.reshape(1, -1)

# --- Recurrence Plot ---
transformer = RecurrencePlot()
X_rp = transformer.transform(x_recurrence_plot)
rp_img = (X_rp[0] * 255).astype(np.uint8)  # Convert to grayscale image
rp_img_color = cv2.applyColorMap(rp_img, cv2.COLORMAP_JET)

# --- Correlation Signal Image ---
# Normalize to 0-255 for image display
corr_norm = cv2.normalize(x_corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
corr_img = np.tile(corr_norm, (256, 1))  # Repeat to make 2D "image"
corr_img_color = cv2.applyColorMap(corr_img, cv2.COLORMAP_JET)

# --- Display or Save ---
window_title = f'Correlation and Recurrence Plot (Label: {data[pre_idx, -1]})'
cv2.imshow(window_title, rp_img_color)
cv2.imshow(window_title, corr_img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # --- Save Image ---
# image_label = str(data[pre_idx, -1])
# filename = f"combined_corr_rp_{image_label}.png"
# filepath = os.path.join(image_dir, filename)
# cv2.imwrite(filepath, combined)
# print(f"Combined image saved as '{filepath}'")
