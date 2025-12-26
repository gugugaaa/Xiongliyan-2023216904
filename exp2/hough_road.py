import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed by `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    # 支持多通道图像遮罩
    match_mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, roi_vertices=None, color=[0, 255, 0], thickness=10):
    """
    Draws lines on an image.
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    # 使用 ROI mask 对生成的线段进行截断
    if roi_vertices is not None:
        line_img = region_of_interest(line_img, roi_vertices)
    
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

def filter_and_merge_lines(lines, img_shape):
    if lines is None: return None
    left_lane, right_lane = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999
            angle = np.abs(np.arctan(slope) * 180 / np.pi)
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            # 2. 角度阈值过滤 (20-80度) 和 更大长度阈值过滤 (>60)
            if 20 < angle < 80 and length > 60:
                intercept = y1 - slope * x1
                if slope < 0: left_lane.append((slope, intercept))
                else: right_lane.append((slope, intercept))
    
    # 3. 平均合并靠近的一组线
    def average_lane(lane_data):
        if not lane_data: return None
        avg = np.mean(lane_data, axis=0)
        y1 = img_shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - avg[1]) / avg[0])
        x2 = int((y2 - avg[1]) / avg[0])
        return [x1, y1, x2, y2]

    res = []
    l = average_lane(left_lane)
    if l: res.append([l])
    r = average_lane(right_lane)
    if r: res.append([r])
    return res if res else None

def lane_detection_pipeline(frame):
    # 1. Grayscale & Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 3. ROI Mask
    height, width = frame.shape[:2]
    roi_pts = np.array([[(width*0.1, height), (width*0.5, height*0.6), (width*0.6, height*0.6), (width*1.1, height)]], np.int32)
    masked_edges = region_of_interest(edges, roi_pts)

    # 4. Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)

    # 5. Filter and Merge
    merged_lines = filter_and_merge_lines(lines, frame.shape)

    # 6. Draw (传入 ROI 顶点进行截断)
    return draw_lines(frame, merged_lines, roi_pts)

# --- RUN THE CODE ---
if __name__ == "__main__":
    # 1. 处理 road.mp4
    cap = cv2.VideoCapture('properties/road.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 视频播放结束时重置到第一帧实现循环
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        result = lane_detection_pipeline(frame)
        
        cv2.imshow('Lane Detection', result)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()