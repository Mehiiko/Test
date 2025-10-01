import cv2
import numpy as np
import os

# Folder wejściowy i wyjściowy
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Parametry
RADIUS = 780
STEP = 1
ANGLES = 2880
SEARCH_SHIFT = int(input("Podaj wartość SEARCH_SHIFT: "))

# Funkcja dopasowania okręgu
def process_image(img_path, radius=RADIUS):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    angles = np.linspace(0, 2*np.pi, ANGLES, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    def score_center(cx, cy, r=radius):
        xs = (cx + r * cos_a).astype(np.int32)
        ys = (cy + r * sin_a).astype(np.int32)
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(mask):
            return -1e9
        return float(grad_mag[ys[mask], xs[mask]].mean())

    center0 = (w//2, h//2)
    best_center = center0
    best_score = score_center(*center0)

    for dy in range(-SEARCH_SHIFT, SEARCH_SHIFT+1, STEP):
        for dx in range(-SEARCH_SHIFT, SEARCH_SHIFT+1, STEP):
            cx = center0[0] + dx
            cy = center0[1] + dy
            sc = score_center(cx, cy)
            if sc > best_score:
                best_score, best_center = sc, (cx, cy)

    # Maska
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, best_center, radius, 255, -1)

    plate_black = cv2.bitwise_and(img, img, mask=mask)
    plate_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    plate_rgba[:, :, 3] = mask

    # Przycięcie do bounding boxa
    x_min = max(best_center[0] - radius, 0)
    x_max = min(best_center[0] + radius, img.shape[1])
    y_min = max(best_center[1] - radius, 0)
    y_max = min(best_center[1] + radius, img.shape[0])
    crop_alpha = plate_rgba[y_min:y_max, x_min:x_max]

    return crop_alpha, best_center, radius

# Przetwarzanie wszystkich plików w folderze
for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    in_path = os.path.join(input_dir, fname)
    stem = os.path.splitext(fname)[0]

    crop_alpha, center, radius = process_image(in_path)

    out_alpha = os.path.join(output_dir, f"{stem}_crop.png")

    cv2.imwrite(out_alpha, crop_alpha)

    print(f"✔ {fname}: center={center}, radius={radius}")
