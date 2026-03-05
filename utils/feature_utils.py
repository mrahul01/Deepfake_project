import cv2
import numpy as np


# ---------------------------
# Gabor Preprocessing
# ---------------------------
def gabor_preprocess(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getGaborKernel(
        (9, 9),
        4.0,
        np.pi/4,
        10.0,
        0.5
    )

    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)

    return filtered


# ---------------------------
# Color Statistics
# ---------------------------
def color_stats(frame):

    stats = []

    for i in range(3):

        channel = frame[:,:,i]

        stats.append(np.mean(channel))
        stats.append(np.std(channel))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(3):

        channel = hsv[:,:,i]

        stats.append(np.mean(channel))
        stats.append(np.std(channel))

    return stats


# ---------------------------
# Luminance
# ---------------------------
def luminance_mean_std(frame):

    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    y = yuv[:,:,0]

    return np.mean(y), np.std(y)


# ---------------------------
# Gray Histogram
# ---------------------------
def gray_histogram(img, bins=32):

    hist = cv2.calcHist([img],[0],None,[bins],[0,256])

    hist = hist.flatten()

    hist = hist / np.sum(hist)

    return hist


# ---------------------------
# Edge Density
# ---------------------------
def edge_density(img):

    edges = cv2.Canny(img.astype(np.uint8),50,150)

    return np.mean(edges > 0)


# ---------------------------
# DCT Energy
# ---------------------------
def dct_energy_features(img):

    img = cv2.resize(img,(128,128)).astype(np.float32)

    dct = cv2.dct(img)

    total_energy = np.sum(dct**2)

    low = dct[:16,:16]

    low_energy = np.sum(low**2)

    high_ratio = (total_energy - low_energy) / total_energy

    return total_energy, high_ratio


# ---------------------------
# Fuzzy C Means
# ---------------------------
def fuzzy_c_means_intensity(img, c=3, iters=8):

    x = img.flatten().astype(np.float32)

    centers = np.random.choice(x, c)

    for _ in range(iters):

        dist = np.abs(x[:,None] - centers)

        membership = 1 / (dist + 1e-6)

        membership = membership / membership.sum(axis=1, keepdims=True)

        centers = (membership * x[:,None]).sum(axis=0) / membership.sum(axis=0)

    return np.sort(centers)


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features_from_frame(frame, cfg):

    pre = gabor_preprocess(frame)

    rgb_hsv = color_stats(frame)

    lum_mean, lum_std = luminance_mean_std(frame)

    hist = gray_histogram(pre)

    var = np.var(pre)

    edge = edge_density(pre)

    dct_total, dct_ratio = dct_energy_features(pre)

    feats = np.array(
        rgb_hsv + [lum_mean, lum_std, var, edge, dct_total, dct_ratio],
        dtype=np.float32
    )

    feats = np.concatenate([feats, hist])

    if cfg["use_fcm"]:

        centers = fuzzy_c_means_intensity(pre, c=cfg["fcm_k"])

        feats = np.concatenate([feats, centers])

    return feats
