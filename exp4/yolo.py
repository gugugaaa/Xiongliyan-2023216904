import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Grad-CAM工具类
class YOLOGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()

    def generate(self, img_tensor, class_idx=1):
        self.model.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
        preds = self.model.model(img_tensor)
        output = preds[0] 
        target_class_idx = 4 + class_idx 
        target_score = output[0, target_class_idx, :].max()
        self.model.zero_grad()
        target_score.backward()
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-7)
        return cam

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolo11n.pt') 

img_path = 'bike.png'
results = model.predict(img_path, classes=[1], conf=0.7, save=False)
res_plotted = results[0].plot() 
img_with_box = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

target_layer = model.model.model[-2] 
print(f"Target Layer: {target_layer}")

grad_cam = YOLOGradCAM(model, target_layer)

img_raw = cv2.imread(img_path)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img_h, img_w = img_raw.shape[:2]
input_tensor = cv2.resize(img_raw, (640, 640))
input_tensor = input_tensor.astype(np.float32) / 255.0
input_tensor = input_tensor.transpose(2, 0, 1)
input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
input_tensor.requires_grad = True

try:
    heatmap = grad_cam.generate(input_tensor, class_idx=1)
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_img = (0.5 * img_raw + 0.5 * heatmap_colored).astype(np.uint8)
except Exception as e:
    print(f"Error generating heatmap: {e}")
    overlay_img = img_raw
finally:
    grad_cam.remove_hooks()

# 可视化：左为检测框，右为热力图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Detected Output (Box)")
plt.imshow(img_with_box)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Class Activation Map (Target: Bicycle)")
plt.imshow(overlay_img)
plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('yolo11_bike_experiment.png')