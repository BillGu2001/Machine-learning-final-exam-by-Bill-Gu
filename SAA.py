import matplotlib.pyplot as plt
import math
import random


# 绘制最优路径
def draw_bestway(x, y):
    best_x = []
    best_y = []

    for i in range(n):
        p = bestway[i]
        best_x.append(x[p])
        best_y.append(y[p])
    best_x.append(best_x[0])
    best_y.append(best_y[0])
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title("TSP图解")
    plt.plot(best_x, best_y, color="green", linestyle="-", marker="o", markerfacecolor="red")
    plt.show()


# 计算最优路径分值
def comp_bestway_score(x, y):
    best_x = []
    best_y = []

    for i in range(n):
        p = bestway[i]
        best_x.append(x[p])
        best_y.append(y[p])
    value = 0.0
    for i in range(1, n):
        x2 = (best_x[i - 1] - best_x[i]) * (best_x[i - 1]-best_x[i])
        y2 = (best_y[i - 1] - best_y[i]) * (best_y[i - 1] - best_y[i])
        value = value + math.sqrt(x2 + y2)
    x2 = (best_x[0] - best_x[n - 1]) * (best_x[0] - best_x[n - 1])
    y2 = (best_y[0] - best_y[n - 1]) * (best_y[0] - best_y[n - 1])
    value = value + math.sqrt(x2 + y2)
    return value


# 建立距离矩阵
def build_dis_array(x, y):
    for i in range(n):
        for j in range(n):
            dis_array[i][j] = math.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))


# 打印距离矩阵
def print_dis_array():
    for i in range(n):
        for j in range(n):
            print('%.2f' % dis_array[i][j], end=',')
        print()


# 模拟退火算法
# 利用温度控制逐步减少接受差解的概率，最终达到局部最优
def saa(x, y):
    obway = [0] * n  # 保存当前全局最优解
    t = 20
    tf = 0.001
    obscore = comp_bestway_score(x, y)  # obscore保存全局最优值
    for i in range(n):
        obway[i] = bestway[i]
    f1 = obscore  # f1保存当前解分值
    while t > tf:
        for iter in range(10):  # iter为等温过程迭代次数
            v1 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if v1 != v2:  # 随机选择两个不同的结点
                bestway[v1], bestway[v2] = bestway[v2], bestway[v1]  # 交换随机选择的两个节点
                f2 = comp_bestway_score(x, y)  # 计算候选解的分值，f2保存候选解
                if f2 < obscore:  # 如果候选解分值优于全局最优解，则更新全局最优解
                    obscore = f2
                    for i in range(n):
                        obway[i] = bestway[i]
                delta = f2 - f1  # 计算候选解与当前解之间的差值
                alpha = random.random()
                if delta < 0:  # 直接接受候选解
                    f1 = f2
                elif alpha < math.exp(-delta / t):  # 根据Metropolis准则接受交换
                    f1 = f2
                else:  # 不接受候选解，回退到当前解
                    bestway[v2], bestway[v1] = bestway[v1], bestway[v2]
        t = t * 0.995  # 降温


if __name__ == '__main__':
    x = [0, 1, 5, 3, 5]
    y = [0, 4, 5, 3, 2]
    n = len(x)
    bestway = [i for i in range(n)]
    dis_array = [[0] * n for i in range(n)]

    build_dis_array(x, y)
    print_dis_array()
    print("\nsa:")
    saa(x, y)
    print("bestway", bestway)
    print("score:", comp_bestway_score(x, y))
    draw_bestway(x, y)

