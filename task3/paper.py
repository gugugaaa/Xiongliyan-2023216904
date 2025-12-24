import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 替换原有硬编码路径为可配置参数
parser = argparse.ArgumentParser(description="SIFT 特征匹配示例")
parser.add_argument("--data_dir", type=str, default="tsukuba", help="包含 ppm/pgm/png 的目录")
parser.add_argument("--left", type=str, default="scene1.row3.col3.ppm", help="左目文件名")
parser.add_argument("--right", type=str, default="scene1.row3.col4.ppm", help="右目文件名")

# 新增可控参数
parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio 阈值 (knn 模式)")
parser.add_argument("--use_ransac", action="store_true", help="是否使用 RANSAC 进行几何验证")
parser.add_argument("--ransac_thresh", type=float, default=3.0, help="RANSAC 重投影阈值（像素）")
parser.add_argument("--max_display", type=int, default=25, help="最多绘制的匹配数量（采样）")
parser.add_argument("--cross_check", action="store_true", help="使用 BFMatcher(crossCheck=True)")
parser.add_argument("--seed", type=int, default=42, help="随机种子（用于采样可复现）")
parser.add_argument("--save", type=str, default="", help="保存可视化图像到文件（若为空则不保存）")
# 新增 gap 参数（两图之间的像素间隙）
parser.add_argument("--gap", type=int, default=20, help="两图之间的像素间隙（像素，>=0）")
args = parser.parse_args()

# 支持相对路径（相对于脚本）或绝对路径
base_dir = args.data_dir
if not os.path.isabs(base_dir):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), base_dir))

img1_path = os.path.join(base_dir, args.left)
img2_path = os.path.join(base_dir, args.right)

print(f"使用图像：\n  左：{img1_path}\n 右：{img2_path}")
print(f"两图间隙: {args.gap} px")
# 读取图片
img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

if img1 is None or img2 is None:
    print("错误：无法找到图片，请检查路径和文件名。")
else:
    # 如果图像为多通道则转换为灰度；如果是单通道（如 pgm），直接使用
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()

    # 如果需要在两张图之间留空隙，则对左图在右侧填充一条宽为 gap 的常量条带（灰色）
    gap = max(0, int(args.gap))
    if gap > 0:
        # 对彩色图像与灰度图像同时填充，确保关键点坐标与图像一致
        if len(img1.shape) == 3:
            img1 = cv2.copyMakeBorder(img1, 0, 0, 0, gap, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
        else:
            img1 = cv2.copyMakeBorder(img1, 0, 0, 0, gap, borderType=cv2.BORDER_CONSTANT, value=255)
        gray1 = cv2.copyMakeBorder(gray1, 0, 0, 0, gap, borderType=cv2.BORDER_CONSTANT, value=255)

    # -----------------------------------------------------------
    # 步骤 1 & 2: SIFT 特征检测与描述子计算
    # -----------------------------------------------------------
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"在图1中检测到 {len(kp1)} 个关键点")
    print(f"在图2中检测到 {len(kp2)} 个关键点")

    # 检查描述子是否存在
    if des1 is None or des2 is None:
        print("错误：未能计算到描述子（des1 或 des2 为空）。请检查图像质量或更换图片。")
    else:
        # -----------------------------------------------------------
        # 步骤 3: 特征匹配（支持 crossCheck 或 knn+ratio）
        # -----------------------------------------------------------
        good_matches = []
        if args.cross_check:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            raw_matches = bf.match(des1, des2)
            raw_matches = sorted(raw_matches, key=lambda x: x.distance)
            good_matches = raw_matches  # 若需要可在此按距离裁剪
            print(f"使用 crossCheck，初始匹配: {len(good_matches)}")
        else:
            bf = cv2.BFMatcher()
            knn_matches = bf.knnMatch(des1, des2, k=2)
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < args.ratio * n.distance:
                    good_matches.append(m)
            print(f"knn+ratio (ratio={args.ratio}) 筛选后匹配: {len(good_matches)}")

        # 如果匹配数量太少，直接退出或跳过 RANSAC
        if len(good_matches) == 0:
            print("未找到有效匹配，停止。")
        else:
            # -----------------------------------------------------------
            # 步骤 4: 可选的几何验证 (RANSAC)
            # -----------------------------------------------------------
            inlier_matches = good_matches
            if args.use_ransac:
                if len(good_matches) >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, args.ransac_thresh)
                    if mask is not None:
                        mask = mask.ravel().tolist()
                        inlier_matches = [m for m, v in zip(good_matches, mask) if v]
                        print(f"RANSAC (thresh={args.ransac_thresh}) 保留内点: {len(inlier_matches)}/{len(good_matches)}")
                    else:
                        print("RANSAC 未返回掩码，保留全部匹配。")
                else:
                    print("匹配数不足以运行 RANSAC（需要 >=4），跳过 RANSAC。")

            # -----------------------------------------------------------
            # 步骤 5: 采样用于可视化并绘制
            # -----------------------------------------------------------
            rng = np.random.default_rng(args.seed)
            display_matches = inlier_matches
            if len(display_matches) > args.max_display:
                idxs = rng.choice(len(display_matches), size=args.max_display, replace=False)
                display_matches = [display_matches[i] for i in idxs]
                print(f"随机采样用于绘制: {len(display_matches)} / {len(inlier_matches)}")

            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, display_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                          matchColor=(0, 255, 0),
                                          singlePointColor=None)

            img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(15, 10))
            plt.imshow(img_matches_rgb)
            plt.title(f"SIFT matches")
            plt.axis('off')
            plt.show()

            if args.save:
                # 将 RGB 转回 BGR 保存（OpenCV 写入为 BGR）
                save_bgr = cv2.cvtColor(img_matches_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(args.save, save_bgr)
                print(f"已保存可视化图像到: {args.save}")