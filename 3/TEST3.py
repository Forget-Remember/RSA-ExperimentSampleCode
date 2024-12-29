import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 步骤（1）
# 生成一组 [-1, 1] 均匀分布的随机变量，具有3000样本
samples_1 = np.random.uniform(-1, 1, 3000)

# 绘制柱形图并拟合概率密度曲线
plt.figure(figsize=(12, 5))
count, bins, ignored = plt.hist(
    samples_1, bins=50, density=True, alpha=0.6, color="g", label="样本直方图"
)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (bins[1] - bins[0]) * count.max()
uniform_pdf = p * np.ones_like(bins)
plt.plot(bins, uniform_pdf, "k--", linewidth=2, label="均匀分布PDF")
plt.title("一组 [-1, 1] 均匀分布的随机变量")
plt.xlabel("值")
plt.ylabel("概率密度")
plt.legend()
plt.savefig("一组均匀分布.png")
plt.show()

# 生成第二组 [-1, 1] 均匀分布的随机变量，与第一组相互独立，同样具有3000样本
samples_2 = np.random.uniform(-1, 1, 3000)

# 将两组样本相叠加
sum_samples_2 = samples_1 + samples_2

# 绘制柱形图并拟合概率密度曲线
plt.figure(figsize=(12, 5))
count, bins, ignored = plt.hist(
    sum_samples_2, bins=50, density=True, alpha=0.6, color="b", label="样本直方图"
)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
mu_sum_2 = np.mean(sum_samples_2)
sigma_sum_2 = np.std(sum_samples_2)
normal_pdf = norm.pdf(x, mu_sum_2, sigma_sum_2)
plt.plot(x, normal_pdf, "r-", linewidth=2, label="正态分布拟合")
plt.title("两组 [-1, 1] 均匀分布的随机变量之和")
plt.xlabel("值")
plt.ylabel("概率密度")
plt.legend()
plt.savefig("两组均匀分布之和.png")
plt.show()


# 生成 K 组 [-1, 1] 相互独立的均匀分布的随机变量，每组具有3000样本
def generate_and_plot_sum(K):
    sum_samples = np.zeros(3000)
    for _ in range(K):
        sum_samples += np.random.uniform(-1, 1, 3000)

    # 绘制柱形图并拟合概率密度曲线
    plt.figure(figsize=(12, 5))
    count, bins, ignored = plt.hist(
        sum_samples, bins=50, density=True, alpha=0.6, color="c", label="样本直方图"
    )
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mu_sum_K = np.mean(sum_samples)
    sigma_sum_K = np.std(sum_samples)
    normal_pdf = norm.pdf(x, mu_sum_K, sigma_sum_K)
    plt.plot(x, normal_pdf, "m-", linewidth=2, label="正态分布拟合")
    plt.title(f"{K} 组 [-1, 1] 均匀分布的随机变量之和")
    plt.xlabel("值")
    plt.ylabel("概率密度")
    plt.legend()
    plt.savefig(f"{K}_组均匀分布之和.png")
    plt.show()


# 分别考证 K=4 和 K=7 的情况
generate_and_plot_sum(4)
generate_and_plot_sum(7)

# 绘制标准正态分布的概率密度曲线
plt.figure(figsize=(12, 5))
x = np.linspace(-10, 10, 1000)
standard_normal_pdf = norm.pdf(x, 0, 1)
plt.plot(x, standard_normal_pdf, "k-", linewidth=2, label="标准正态分布PDF")
plt.title("标准正态分布 PDF")
plt.xlabel("值")
plt.ylabel("概率密度")
plt.legend()
plt.savefig("标准正态分布.png")
plt.show()
