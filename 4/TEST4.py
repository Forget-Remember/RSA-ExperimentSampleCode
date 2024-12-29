import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 4096
mean_1 = 0
std_dev_1 = 1
mean_2 = 1.5
std_dev_2 = np.sqrt(2)


# 生成高斯序列并分析
def analyze_gaussian_sequence(mean, std_dev, N, title_suffix):
    # 产生高斯序列
    samples = np.random.normal(mean, std_dev, N)

    # 绘制时域波形
    plt.figure(figsize=(12, 5))
    plt.plot(samples)
    plt.title(f"高斯序列时域波形 ({title_suffix})")
    plt.xlabel("样本索引")
    plt.ylabel("幅度")
    plt.savefig(f"高斯序列时域波形_{title_suffix}.png")
    plt.show()

    # 绘制一维概率密度
    density = gaussian_kde(samples)
    xs = np.linspace(-5, 5, 200)
    plt.figure(figsize=(12, 5))
    plt.plot(xs, density(xs), label="估计的PDF")
    plt.plot(xs, norm.pdf(xs, mean, std_dev), label="理论PDF")
    plt.title(f"一维概率密度 ({title_suffix})")
    plt.legend()
    plt.savefig(f"一维概率密度_{title_suffix}.png")
    plt.show()

    # 绘制二维概率密度（前100个样本）
    plt.figure(figsize=(12, 5))
    plt.hexbin(samples[:-1], samples[1:], gridsize=30, cmap="Blues")
    cb = plt.colorbar(label="每个区间中的计数")
    plt.title(f"二维概率密度 ({title_suffix})")
    plt.xlabel("当前样本")
    plt.ylabel("下一个样本")
    plt.savefig(f"二维概率密度_{title_suffix}.png")
    plt.show()

    return samples


# （1）分析第一个高斯序列
samples_1 = analyze_gaussian_sequence(mean_1, std_dev_1, N, "均值0方差1")


# 计算自相关函数
def autocorr(x):
    result = np.correlate(x, x, mode="full")
    max_corr = len(x) - 1  # 中心位置用于归一化
    return result[max_corr:] / result[max_corr]


# （2）计算和比较自相关函数
acf_1 = autocorr(samples_1)

# 绘制自相关函数
lags = np.arange(N)
plt.figure(figsize=(12, 5))
plt.plot(lags, acf_1)
plt.title("自相关函数 (均值0方差1)")
plt.xlabel("滞后")
plt.ylabel("自相关")
plt.grid(True)
plt.savefig("自相关函数_均值0方差1.png")
plt.show()

# 理论上的自相关函数对于独立同分布的高斯随机变量只在零滞后处有一个峰值等于方差，在其他位置则为零。
# 对于仿真结果与理论结果的不同，这可能是由于有限的样本数量导致的统计波动。

# （3）分析第二个高斯序列
samples_2 = analyze_gaussian_sequence(mean_2, std_dev_2, N, "均值1.5方差2")
acf_2 = autocorr(samples_2)

# 比较两个自相关函数的结果
plt.figure(figsize=(12, 5))
plt.plot(lags, acf_1, label="均值0, 方差1")
plt.plot(lags, acf_2, label="均值1.5, 方差2")
plt.title("自相关函数比较")
plt.xlabel("滞后")
plt.ylabel("自相关")
plt.legend()
plt.grid(True)
plt.savefig("自相关函数比较.png")
plt.show()

# 序列的均值和方差影响自相关函数的方式：
# 均值的变化不会影响自相关函数的形状，因为自相关是基于样本减去其平均值后的值计算的。
# 方差的变化会影响自相关函数的高度，因为自相关函数的值是相对于方差归一化的。
