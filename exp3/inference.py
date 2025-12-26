import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 必须重新定义一遍模型结构以加载权重
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_digit(img_roi):
    """
    将切割出来的数字图像处理成类似 MNIST 的格式 (28x28, 黑底白字, 居中)
    """
    # 1. 调整大小，保持长宽比
    h, w = img_roi.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    
    resized = cv2.resize(img_roi, (new_w, new_h))
    
    # 2. 填充到 28x28 (Padding)
    pad_img = np.zeros((28, 28), dtype=np.uint8)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    pad_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    # 3. 转换为 PIL Image 并归一化
    pil_img = Image.fromarray(pad_img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(pil_img).unsqueeze(0) # 增加 batch 维度

def predict_student_id(image_path, model_path):
    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 读取图像并进行 OpenCV 预处理
    img_original = cv2.imread(image_path)
    if img_original is None:
        print("Error: Image not found.")
        return

    # 转灰度
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (Otsu's thresholding)
    # 注意：如果是白纸黑字，用 cv2.THRESH_BINARY_INV 进行反转，变成黑底白字
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤太小的噪点轮廓，并按从左到右排序
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100: # 面积阈值，根据实际情况调整
            digit_contours.append((x, y, w, h, cnt))
    
    # 按 x 坐标排序 (保证识别顺序是从左到右)
    digit_contours.sort(key=lambda x: x[0])

    result_string = ""
    plt.figure(figsize=(10, 4))
    
    print("Recognizing digits...")
    
    for i, (x, y, w, h, cnt) in enumerate(digit_contours):
        # 提取 ROI (Region of Interest)
        roi = thresh[y:y+h, x:x+w]
        
        # 预处理 ROI 变成 Tensor
        img_tensor = preprocess_digit(roi).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        
        result_string += str(pred)
        
        # 可视化：画框并标记
        cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_original, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 显示最终结果
    print(f"Final Recognized Student ID: {result_string}")
    
    # Matplotlib 显示 (OpenCV BGR 转 RGB)
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Recognition Result: {result_string}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 替换成你的照片文件名
    predict_student_id('my_id.png', 'exp2/mnist_cnn.pth')