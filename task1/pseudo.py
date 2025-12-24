# 构建代价体
function BuildCostVolume(IL, IR, Dmax, winRadius):
    for y = 0..H-1:
        for x = 0..W-1:
            for d = 0..Dmax:
                xr = x - d
                if xr < 0:
                    C[y,x,d] = +INF
                else:
                    C[y,x,d] = PatchCost(IL, IR, y, x, xr, winRadius)
    return C

# 对每条扫描线做DP
function ScanlineDP(C, y, Dmax, OccCost):
    N = W   # 该行长度
    # DP 表：M[i][j] = 最小总代价，i/j 从 0..N
    init M[0..N][0..N] = +INF
    init P[0..N][0..N] = NONE   # 记录回溯指针
    M[0][0] = 0

    for i = 0..N:
        for j = 0..N:
            if i>0 and j>0:
                d = i - j
                if 0 <= d <= Dmax:
                    cost_match = M[i-1][j-1] + C[y, x=i-1, d]
                    Relax(M, P, i, j, cost_match, FROM_MATCH)

            if i>0:
                cost_occL = M[i-1][j] + OccCost
                Relax(M, P, i, j, cost_occL, FROM_OCCLUDE_LEFT)

            if j>0:
                cost_occR = M[i][j-1] + OccCost
                Relax(M, P, i, j, cost_occR, FROM_OCCLUDE_RIGHT)

    # 回溯得到该行视差
    DispRow[0..N-1] = INVALID
    i = N; j = N
    while i>0 or j>0:
        move = P[i][j]
        if move == FROM_MATCH:
            d = i - j
            xL = i - 1
            DispRow[xL] = d
            i = i - 1; j = j - 1
        else if move == FROM_OCCLUDE_LEFT:
            xL = i - 1
            DispRow[xL] = INVALID   # 或标记为遮挡
            i = i - 1
        else if move == FROM_OCCLUDE_RIGHT:
            j = j - 1

    return DispRow

# 整幅图执行 + 后处理
function StereoCorrespondence_DP(IL, IR, Dmax):
    # 代价体
    C = BuildCostVolume(IL, IR, Dmax, winRadius=r)

    # 每行DP
    for y = 0..H-1:
        DispL[y,*] = ScanlineDP(C, y, Dmax, OccCost=lambda)
 
    # 后处理
    DispL = FillInvalidByNeighbor(DispL)
    DispL = MedianFilter(DispL, k=3)

    return DispL
