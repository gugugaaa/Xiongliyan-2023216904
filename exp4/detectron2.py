import os
import cv2
import torch
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import timm
import torchvision
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ==========================================
# 1. 工具函数
# ==========================================

def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    d = x.max() - x.min()
    return x / (d + 1e-8)

def _preprocess_imagenet(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

# ==========================================
# 2. 核心组合函数
# ==========================================

def run_combined_analysis(image_path, det2_layer='p3', gradcam_layer='layer4'):
    """
    Args:
        image_path: 图片路径或URL
        det2_layer: Detectron2 Backbone提取的层 (通常为 p2, p3, p4, p5)
        gradcam_layer: timm ResNet50 挂载 Grad-CAM 的层 (通常为 layer1, layer2, layer3, layer4)
    """
    # --- 准备工作 ---
    if image_path.startswith("http"):
        local_img = "input_image.png"
        if not os.path.exists(local_img):
            urllib.request.urlretrieve(image_path, local_img)
        image_path = local_img

    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 流程一：Detectron2 特征提取 & 目标识别
    # ==========================================
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)

    # 挂载 Hook 提取特征
    det2_features = {}
    def det2_hook(module, inp, out):
        det2_features['feat'] = out

    # Detectron2 的 backbone 通常输出 dict {'p2':..., 'p3':...}
    handle_det2 = predictor.model.backbone.register_forward_hook(det2_hook)
    
    # 推理
    outputs = predictor(bgr)
    handle_det2.remove()

    # 处理左图：特征提取强度 (从指定层提取)
    # 注意：Detectron2 的 FPN 输出是字典，我们按 det2_layer 索引
    feat_map = det2_features['feat'][det2_layer].cpu().detach().numpy()[0]
    intensity_map = np.mean(feat_map, axis=0)
    intensity_map = _norm01(intensity_map)

    # 处理右图：目标检测识别结果 (仅筛选自行车)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes
    instances = outputs["instances"].to("cpu")
    bike_indices = [i for i, label in enumerate(instances.pred_classes) if class_names[label] == 'bicycle']
    bicycle_instances = instances[bike_indices]
    
    v = Visualizer(rgb, metadata=metadata, scale=1.0)
    det_result = v.draw_instance_predictions(bicycle_instances).get_image()

    # ==========================================
    # 流程二：timm ResNet50 Grad-CAM
    # ==========================================
    model_t = timm.create_model("resnet50", pretrained=True).to(device).eval()
    
    activations = {}
    gradients = {}
    def fwd_hook(module, inp, out): activations["feat"] = out
    def bwd_hook(grad): gradients["grad"] = grad

    # 挂载到指定层，例如 'layer4'
    target_layer = getattr(model_t, gradcam_layer)
    handle_fwd = target_layer.register_forward_hook(fwd_hook)

    # 前向计算
    input_tensor = _preprocess_imagenet(rgb).to(device)
    with torch.set_grad_enabled(True):
        logits = model_t(input_tensor)
        # 寻找 ImageNet 中与自行车相关的类别
        categories = torchvision.models.ResNet50_Weights.DEFAULT.meta["categories"]
        bike_keywords = ["mountain bike", "bicycle-built-for-two", "tricycle"]
        target_idx = 671 # 默认使用 mountain bike 的 ID
        for i, cat in enumerate(categories):
            if any(k in cat.lower() for k in bike_keywords):
                target_idx = i
                break
        
        target_logit = logits[0, target_idx]
        
        # 反向传播获取梯度
        model_t.zero_grad()
        # 为获取梯度，给激活值注册 hook
        handle_bwd = activations["feat"].register_hook(bwd_hook)
        target_logit.backward()
        
    # 计算 Grad-CAM
    weights = gradients["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["feat"]).sum(dim=1)).squeeze(0)
    cam_np = cam.detach().cpu().numpy()
    cam_up = cv2.resize(_norm01(cam_np), (rgb.shape[1], rgb.shape[0]))

    # 移除所有 Hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # ==========================================
    # 3. 最终可视化
    # ==========================================
    plt.figure(figsize=(24, 8))

    # 左图：特征强度 (Backbone)
    plt.subplot(1, 3, 1)
    plt.imshow(intensity_map, cmap='viridis')
    plt.title(f"1. Feature Intensity\n(Detectron2 Backbone: {det2_layer})", fontsize=14)
    plt.axis('off')

    # 中图：Grad-CAM (timm)
    plt.subplot(1, 3, 2)
    plt.imshow(rgb)
    plt.imshow(cam_up, cmap='jet', alpha=0.5) # 叠加显示
    plt.title(f"2. Grad-CAM Heatmap\n(timm {gradcam_layer} for '{categories[target_idx]}')", fontsize=14)
    plt.axis('off')

    # 右图：识别结果
    plt.subplot(1, 3, 3)
    plt.imshow(det_result)
    plt.title("3. Detection Result\n(Faster R-CNN: Bboxes)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("combined_analysis.jpg", dpi=300)
    plt.show()
    print("分析完成，结果已保存为 combined_analysis.jpg")

# ==========================================
# 4. 执行入口 (在这里修改层参数)
# ==========================================
if __name__ == "__main__":
    # 在这里直接修改你想查看的层
    # Detectron2 可选: 'p2', 'p3', 'p4', 'p5'
    # Grad-CAM 可选: 'layer1', 'layer2', 'layer3', 'layer4'
    run_combined_analysis(
        image_path="https://i.ibb.co/DPgB5w3j/bike.png",
        det2_layer='p3', 
        gradcam_layer='layer4'
    )