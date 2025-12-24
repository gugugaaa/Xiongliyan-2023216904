import numpy as np
import cv2
import argparse
import os
import maxflow  # 需安装 PyMaxflow: pip install PyMaxflow
import matplotlib.pyplot as plt

def compute_data_cost(imgL, imgR, dmax, block_size):
    """
    计算数据代价 (Data Cost Volume)
    使用块匹配 SAD (Sum of Absolute Differences)
    """
    h, w = imgL.shape
    # 初始化代价体 (H, W, D)
    # PyMaxflow 要求数据项类型通常为 int 或 float，这里为了精度和内存折中用 float32
    # 注意：aexpansion_grid 内部通常处理 int32，需要根据 scale 调整，这里简化直接传
    data_cost = np.full((h, w, dmax), 255.0 * block_size * block_size, dtype=np.float32)
    
    # 将图像转为 float 以便计算
    L = imgL.astype(np.float32)
    R = imgR.astype(np.float32)

    kernel = np.ones((block_size, block_size), np.float32)

    for d in range(dmax):
        # 构造平移后的右图
        if d == 0:
            shifted_R = R
        else:
            shifted_R = np.zeros_like(R)
            shifted_R[:, d:] = R[:, :-d] # 右移 d 个像素
        
        # 计算绝对差
        diff = np.abs(L - shifted_R)
        
        # 使用 boxFilter 聚合窗口代价 (相当于 block matching)
        # normalize=False 使得结果是 SAD 而不是 MAE
        cost_map = cv2.boxFilter(diff, -1, (block_size, block_size), normalize=False)
        
        data_cost[:, :, d] = cost_map

    # 如果 d > 0，左侧 d 列的代价是无效的，保持最大值即可
    # 转换为 PyMaxflow 所需的格式 (通常尽量归一化到整数范围以提高图割效率，这里保持 float)
    return data_cost

def stereo_graph_cuts(data_cost, penalty):
    """
    使用 Alpha-Expansion 算法进行能量最小化
    """
    h, w, n_labels = data_cost.shape
    
    # 1. 构造平滑项 (Pairwise Term) - Potts Model
    # V(l1, l2) = penalty if l1 != l2 else 0
    # 这是一个 (n_labels, n_labels) 的矩阵
    smoothness = np.full((n_labels, n_labels), penalty, dtype=np.float32)
    np.fill_diagonal(smoothness, 0)

    # 2. 调用 PyMaxflow 的 alpha-expansion
    # data_cost 需要 reshape 成 (H*W, n_labels) 或者直接传入 grid
    # fastmin.aexpansion_grid 是专门针对 2D 网格优化的
    print("Running Graph Cuts (Alpha-Expansion)...")
    labels = maxflow.fastmin.aexpansion_grid(
        data_cost.astype(np.float64),  # Unary terms (H, W, L)
        smoothness.astype(np.float64)  # Pairwise terms (L, L)
    )
    
    return labels.reshape(h, w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="task1/tsukuba", help="包含 ppm/pgm/png 的目录")
    parser.add_argument("--left", type=str, default="scene1.row3.col3.ppm", help="左目文件名")
    parser.add_argument("--right", type=str, default="scene1.row3.col4.ppm", help="右目文件名")
    parser.add_argument("--gt", type=str, default="truedisp.row3.col3.pgm", help="GT disparity 文件名（可选）")
    parser.add_argument("--gt_scale", type=float, default=16, help="GT 缩放系数 (如 png 存储通常 * 16)")
    parser.add_argument("--dmax", type=int, default=16, help="最大视差 (Tsukuba通常为16-32)")
    parser.add_argument("--bm_block", type=int, default=3, help="Block size (奇数)")
    parser.add_argument("--dp_penalty", type=float, default=20.0, help="平滑项惩罚系数 (Lambda)")
    args = parser.parse_args()

    # 1. 读取图像
    left_path = os.path.join(args.data_dir, args.left)
    right_path = os.path.join(args.data_dir, args.right)
    gt_path = os.path.join(args.data_dir, args.gt)

    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print(f"Error: Cannot read images from {args.data_dir}")
        exit()

    print(f"Loaded images: {imgL.shape}")

    # 2. 计算 Data Cost Volume
    # 将代价缩小一点以适应 fastmin 的计算 (可选)
    cost_vol = compute_data_cost(imgL, imgR, args.dmax, args.bm_block)
    
    # 3. 运行 Graph Cuts (Energy Minimization)
    disp_map = stereo_graph_cuts(cost_vol, args.dp_penalty)
    
    # 4. 后处理与可视化
    # 归一化视差图以便显示 (0-255)
    disp_vis = (disp_map * (255.0 / args.dmax)).astype(np.uint8)

    # 5. 处理 GT 并拼接
    if os.path.exists(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # GT 通常需要缩放 (例如 Middlebury 数据集视差图通常乘了系数)
        gt_float = gt.astype(np.float32) / args.gt_scale
        gt_vis = (gt_float * (255.0 / args.dmax)).astype(np.uint8)
        
        # 调整大小一致 (有时候 GT 大小可能略有不同)
        if gt_vis.shape != disp_vis.shape:
            gt_vis = cv2.resize(gt_vis, (disp_vis.shape[1], disp_vis.shape[0]))

        # 计算简单的误差 (RMSE)
        mask = gt > 0 # 忽略无 GT 区域
        error = np.abs(disp_map[mask] - gt_float[mask])
        mae = np.mean(error)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

        # 使用 matplotlib 显示并保存 GT 与结果对比
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title('GT')
        axs[0].axis('off')
        axs[1].imshow(disp_vis, cmap='gray', vmin=0, vmax=255)
        axs[1].set_title('GraphCut')
        axs[1].axis('off')
        fig.suptitle(f"GT vs GraphCut | MAE: {mae:.2f}")
        plt.tight_layout()
        plt.savefig("result_graph_cut.png", bbox_inches='tight')
        print("Result saved to result_graph_cut.png")
        plt.show()
    else:
        # 无 GT 时仅显示结果并保存
        plt.figure(figsize=(6, 4))
        plt.imshow(disp_vis, cmap='gray', vmin=0, vmax=255)
        plt.title("Disparity (GraphCut)")
        plt.axis('off')
        plt.savefig("result_graph_cut_no_gt.png", bbox_inches='tight')
        print("GT not found, showing result only. Saved to result_graph_cut_no_gt.png")
        plt.show()