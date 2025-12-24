function DetectScaleInvariantKeypoints(image, num_octaves, num_scales, sigma):
    # 1. 构建高斯尺度空间 (Gaussian Scale Space)
    # 
    gaussian_pyramid = []
    for o from 1 to num_octaves:
        octave_images = []
        for s from 1 to num_scales + 3:
            current_sigma = compute_sigma(o, s, sigma)
            octave_images.append(GaussianBlur(image, current_sigma))
        gaussian_pyramid.append(octave_images)
        image = Downsample(octave_images[num_scales]) # 下采样进入下一阶
        
    # 2. 计算高斯差分金字塔 (Difference of Gaussians - DoG)
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = []
        for i from 1 to len(octave) - 1:
            dog_octave.append(octave[i+1] - octave[i]) # 相邻尺度相减
        dog_pyramid.append(dog_octave)
        
    # 3. 极值点检测 (Local Extrema Detection)
    keypoints = []
    for o from 1 to num_octaves:
        for s from 1 to num_scales:
            current_dog = dog_pyramid[o][s]
            prev_dog = dog_pyramid[o][s-1]
            next_dog = dog_pyramid[o][s+1]
            
            for pixel(x, y) in current_dog:
                # 与 3x3 邻域及上下两个尺度的 18 个邻点（共 26 个邻居）比较
                if is_local_extrema(pixel, prev_dog, current_dog, next_dog):
                    # 4. 关键点精确定位与过滤
                    candidate = RefineLocation(x, y, s) # 泰勒展开亚像素修正
                    if is_stable(candidate): # 剔除低对比度点和边缘响应点
                        keypoints.append(candidate)
                        
    return keypoints