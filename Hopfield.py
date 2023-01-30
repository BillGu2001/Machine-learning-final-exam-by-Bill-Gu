import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, city_position, circle=30, T = 0.1, A = 1.5, D = 1):
        self.city_position = city_position  # 城市位置坐标
        self.n = len(city_position)  # 城市个数
        self.circle = circle  # 循环次数
        self.T = T  # delta_t
        self.d = self.d()  # 距离矩阵
        self.v = self.v()  # 换位矩阵
        self.u0 = 0.2
        self.A = A
        self.D = D

    def d(self):
        d = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    d[i, j] = 0
                else:
                    d[i, j] = np.sqrt((self.city_position[i, 0] - self.city_position[j, 0]) ** 2 +
                                      (self.city_position[i, 1] - self.city_position[j, 1]) ** 2)
        return d

    def tanh(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def v(self):
        a = np.arange(0, self.n)
        np.random.seed(1)
        np.random.shuffle(a)
        v = np.zeros((self.n, self.n))
        for i in range(self.n):
            v[a[i], i] = 1
        return v

    def cal_du_dt(self, v):
        du_dt = np.zeros((self.n, self.n))
        for x in range(self.n):
            for i in range(self.n):
                p1 = 0
                p2 = 0
                p3 = 0
                for j in range(self.n):
                    p1 += v[x, j]
                for y in range(self.n):
                    p2 += v[y, i]
                    if i < self.n - 1:
                        p3 += self.d[x, y] * v[y, i + 1]
                    else:
                        p3 += self.d[x, y] * v[y, 0]
                du_dt[x, i] = -self.A * (p1 - 1) - self.A * (p2 - 1) - self.D * p3
        return du_dt

    def update_u(self, ut, du_dt):
        u = ut + du_dt * self.T
        return u

    def update_v(self, u):
        return (1 + self.tanh(u / self.u0)) / 2

    def energy(self, v, u):
        p1 = 0
        p2 = 0
        p3 = 0
        for x in range(self.n):
            a = 0
            for i in range(self.n):
                a += v[x, i]
            p1 += (a - 1) ** 2
        for i in range(self.n):
            a = 0
            for x in range(self.n):
                a += v[x, i]
            p2 += (a - 1) ** 2
        for x in range(self.n):
            for y in range(self.n):
                for i in range(self.n):
                    if i < self.n - 1:
                        p3 += v[x, i] * self.d[x, y] * v[x, i] * v[y, i + 1]
                    else:
                        p3 += v[x, i] * self.d[x, y] * v[x, i] * v[y, 0]
        e = self.A * p1 / 2 + self.A * p2 / 2 + self.D * p3 / 2
        return e

    def hopfield(self):
        v_0 = self.v
        np.random.seed(0)
        u_0 = np.random.normal(size=(self.n, self.n))  # 初始化V、U
        E = []  # 记录循环过程中网络能量变化
        for t in range(self.circle):
            du_dt = self.cal_du_dt(v_0)
            u = self.update_u(u_0, du_dt)
            v = self.update_v(u)
            u_0 = u
            v_0 = v
            e = self.energy(u=u, v=v)
            E.append(e)
        result = v_0
        return result, E


if __name__ == '__main__':
    city_position = np.array([[0, 0],
                              [1, 4],
                              [5, 5],
                              [3, 3],
                              [5, 2]])
    hopfield = Hopfield(city_position=city_position)
    r, e = hopfield.hopfield()
    print(hopfield.v)

    plt.plot(e)
    plt.xlabel("step", fontsize=15)
    plt.ylabel("energy", fontsize=15)
    plt.grid(linestyle="--")
    plt.show()

