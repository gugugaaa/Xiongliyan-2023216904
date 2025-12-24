function ComputeSAD(IL, IR, xL, yL, d, r):
    """计算两个像素块之间的相似度代价。"""
    sad_score = 0
    for i = -r..r:
        for j = -r..r:
            valL = IL[yL + i, xL + j]
            valR = IR[yL + i, xL + j - d]
            sad_score += abs(valL - valR)
    return sad_score

function LocalStereoMatching(IL, IR, Dmax, winSize):
    """立体匹配主函数。"""
    H, W = GetImageSize(IL)
    r = winSize / 2
    DisparityMap = Array(H, W, init=0)

    for y = r..H-r-1:
        for x = r..W-r-1:
            best_d = 0
            min_cost = +INF
            
            for d = 0..Dmax:
                if (x - d) < r: 
                    break
                
                current_cost = ComputeSAD(IL, IR, x, y, d, r)
                
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_d = d
            
            DisparityMap[y, x] = best_d

    DisparityMap = BasicSimpleFiller(DisparityMap)
    
    return DisparityMap