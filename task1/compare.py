import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_image(filepath):
    # 读取图片（灰度）
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    return img

def load_image_any_to_gray(filepath):
    """读取 PPM/PNG 等，若是彩色则转灰度."""
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_disp_gt(filepath, scale=1.0):
    """读取 GT disparity（常见是 16-bit PGM），并按 scale 缩放到“像素视差”单位."""
    gt = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(f"GT disparity not found at {filepath}")
    gt = gt.astype(np.float32) * float(scale)
    return gt

def _to_multiple_of_16(x: int) -> int:
    return int(np.ceil(x / 16.0) * 16)

# ==========================================
# (a) WTA (Winner-Takes-All) Baseline
# ==========================================
def run_wta_baseline(imgL, imgR, num_disp=64, block_size=15):
    """
    使用 OpenCV 的 StereoBM 实现。
    StereoBM 本质上就是 Block Matching + WTA。
    """
    num_disp = _to_multiple_of_16(int(num_disp))
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity

# ==========================================
# (b) Scanline DP (Simplified Python Implementation)
# ==========================================
def run_scanline_dp(imgL, imgR, max_disp=64, penalty=10):
    """
    单方向扫描线 DP（水平）。
    注意：这里不用 np.roll，避免环绕导致的错误匹配。
    """
    H, W = imgL.shape
    imgL = imgL.astype(np.float32)
    imgR = imgR.astype(np.float32)

    max_disp = int(max_disp)
    cost_vol = np.zeros((H, W, max_disp), dtype=np.float32)

    for d in range(max_disp):
        shifted_R = np.empty_like(imgR)
        if d == 0:
            shifted_R[:] = imgR
        else:
            # x 方向向右平移 d：shifted_R[:, d:] = imgR[:, :-d]
            shifted_R[:, :d] = imgR[:, :1]  # 简单填充左边遮挡区
            shifted_R[:, d:] = imgR[:, :-d]
        cost_vol[:, :, d] = np.abs(imgL - shifted_R)

    dp_vol = np.zeros_like(cost_vol)
    dp_vol[:, 0, :] = cost_vol[:, 0, :]

    P1 = float(penalty)

    print("Running Scanline DP (this may take a moment)...")
    for x in range(1, W):
        prev = dp_vol[:, x - 1, :]

        v0 = prev
        v1 = np.roll(prev, 1, axis=1) + P1
        v1[:, 0] = prev[:, 0] + P1
        v2 = np.roll(prev, -1, axis=1) + P1
        v2[:, -1] = prev[:, -1] + P1

        min_prev = np.minimum(v0, np.minimum(v1, v2))
        dp_vol[:, x, :] = cost_vol[:, x, :] + min_prev

    disp_map = np.argmin(dp_vol, axis=2).astype(np.float32)
    return disp_map

# ==========================================
# (c) StereoSGBM (Semi-Global Block Matching)
# ==========================================
def run_sgbm(imgL, imgR, num_disp=64):
    """
    使用 OpenCV 的 StereoSGBM。
    """
    num_disp = _to_multiple_of_16(int(num_disp))
    window_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disparity

def eval_mae(pred_disp, gt_disp):
    """忽略 gt==0 的无效区."""
    mask = gt_disp > 0
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(pred_disp[mask] - gt_disp[mask])))

def _norm_for_show(d, vmin=0.0, vmax=64.0):
    d = np.asarray(d, dtype=np.float32)
    d = np.clip(d, vmin, vmax)
    return (d - vmin) / max(1e-6, (vmax - vmin))

# ==========================================
# Main Execution & Visualization
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="task1/tsukuba", help="包含 ppm/pgm/png 的目录")
    parser.add_argument("--left", type=str, default="scene1.row3.col3.ppm", help="左目文件名")
    parser.add_argument("--right", type=str, default="scene1.row3.col4.ppm", help="右目文件名")
    parser.add_argument("--gt", type=str, default="truedisp.row3.col3.pgm", help="GT disparity 文件名（可选）")
    parser.add_argument("--gt_scale", type=float, default=1.0, help="GT 缩放系数")
    parser.add_argument("--dmax", type=int, default=64)
    parser.add_argument("--bm_block", type=int, default=15)
    parser.add_argument("--dp_penalty", type=float, default=15.0)
    args = parser.parse_args()

    left_path = os.path.join(args.data_dir, args.left)
    right_path = os.path.join(args.data_dir, args.right)
    gt_path = os.path.join(args.data_dir, args.gt) if args.gt else None

    try:
        imgL = load_image_any_to_gray(left_path)
        imgR = load_image_any_to_gray(right_path)
    except Exception as e:
        print(f"Loading Error: {e}")
        print("Creating dummy synthetic stereo images for demonstration...")
        imgL = np.zeros((288, 384), dtype=np.uint8)
        cv2.rectangle(imgL, (100, 100), (200, 200), 255, -1)
        imgR = np.zeros((288, 384), dtype=np.uint8)
        cv2.rectangle(imgR, (90, 100), (190, 200), 255, -1)

    gt_disp = None
    if gt_path and os.path.exists(gt_path):
        try:
            gt_disp = load_disp_gt(gt_path, scale=args.gt_scale)
        except Exception as e:
            print(f"GT load skipped: {e}")

    DMAX = int(args.dmax)

    disp_wta = run_wta_baseline(imgL, imgR, num_disp=DMAX, block_size=args.bm_block)
    disp_dp = run_scanline_dp(imgL, imgR, max_disp=DMAX, penalty=args.dp_penalty)
    disp_sgbm = run_sgbm(imgL, imgR, num_disp=DMAX)

    if gt_disp is not None:
        print(f"MAE WTA : {eval_mae(disp_wta, gt_disp)}")
        print(f"MAE DP  : {eval_mae(disp_dp, gt_disp)}")
        print(f"MAE SGBM: {eval_mae(disp_sgbm, gt_disp)}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(_norm_for_show(disp_wta, 0, DMAX), cmap="jet")
    plt.title("(a) WTA Baseline (StereoBM)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(_norm_for_show(disp_dp, 0, DMAX), cmap="jet")
    plt.title("(b) Scanline DP (Horizontal Only)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(_norm_for_show(disp_sgbm, 0, DMAX), cmap="jet")
    plt.title("(c) StereoSGBM (OpenCV)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()