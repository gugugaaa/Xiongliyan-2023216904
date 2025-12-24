import cv2
import numpy as np

def detect_sift(img_bgr, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc

def detect_shitomasi(img_bgr, maxCorners=800, qualityLevel=0.01, minDistance=7, blockSize=7):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        blockSize=blockSize,
        useHarrisDetector=False
    )
    kps = []
    if pts is not None:
        for x, y in pts.reshape(-1, 2):
            kps.append(cv2.KeyPoint(float(x), float(y), _size=blockSize))
    return kps, None

def detect_orb(img_bgr, nfeatures=1000, scaleFactor=1.2, nlevels=8):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc

def draw_kps(img_bgr, kps, title="kps"):
    out = cv2.drawKeypoints(img_bgr, kps, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(title, out)
    cv2.waitKey(0)

img = cv2.imread("your.jpg")

k1,_ = detect_sift(img)
k2,_ = detect_shitomasi(img)
k3,_ = detect_orb(img)

draw_kps(img, k1, "SIFT")
draw_kps(img, k2, "Shi-Tomasi")
draw_kps(img, k3, "ORB")
cv2.destroyAllWindows()

