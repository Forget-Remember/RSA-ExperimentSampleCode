import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, skew, kurtosis

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

mean = 0
variance = 2
std_dev = np.sqrt(variance)


# 函数：生成正态分布样本并分析
def analyze_normal_samples(sample_size, title_suffix):
    samples = np.random.normal(mean, std_dev, sample_size)
    sample_mean = np.mean(samples)
    sample_variance = np.var(samples, ddof=1)  # 无偏估计
    print(
        f"样本大小: {sample_size}, 样本均值: {sample_mean:.4f}, 样本方差: {sample_variance:.4f}"
    )

    # 绘制直方图并与正态分布的概率密度函数做比较
    plt.figure(figsize=(12, 5))
    plt.hist(
        samples, bins="auto", density=True, alpha=0.6, color="g", label="样本直方图"
    )
    xs = np.linspace(-6, 6, 200)
    plt.plot(xs, norm.pdf(xs, mean, std_dev), "r-", lw=2, label="理论PDF")
    plt.title(f"正态分布样本 ({title_suffix})")
    plt.xlabel("样本值")
    plt.ylabel("概率密度")
    plt.legend()
    plt.savefig(f"正态分布样本_{sample_size}.png")
    plt.show()

    # 绘制样本的分布图
    plt.figure(figsize=(12, 5))
    plt.plot(samples, "o-", alpha=0.7, label=f"{sample_size}个样本")
    plt.title(f"正态分布样本分布 ({title_suffix})")
    plt.xlabel("索引")
    plt.ylabel("样本值")
    plt.legend()
    plt.savefig(f"正态分布样本分布_{sample_size}.png")
    plt.show()


# 执行（1）和（2）
analyze_normal_samples(25, "25个样本")
analyze_normal_samples(1000, "1000个样本")

# （4）产生均匀分布的样本数据并进行分析
uniform_samples = np.random.uniform(-np.sqrt(3 * variance), np.sqrt(3 * variance), 1000)
uniform_sample_mean = np.mean(uniform_samples)
uniform_sample_variance = np.var(uniform_samples, ddof=1)  # 使用无偏估计
print(
    f"均匀分布样本均值: {uniform_sample_mean:.4f}, 样本方差: {uniform_sample_variance:.4f}"
)

# 绘制直方图并与均匀分布的概率密度函数做比较
plt.figure(figsize=(12, 5))
plt.hist(
    uniform_samples, bins="auto", density=True, alpha=0.6, color="b", label="样本直方图"
)
xs_uniform = np.linspace(-np.sqrt(3 * variance), np.sqrt(3 * variance), 200)
plt.plot(
    xs_uniform,
    uniform.pdf(
        xs_uniform, loc=-np.sqrt(3 * variance), scale=2 * np.sqrt(3 * variance)
    ),
    "r-",
    lw=2,
    label="理论PDF",
)
plt.title("均匀分布样本 (1000个样本)")
plt.xlabel("样本值")
plt.ylabel("概率密度")
plt.legend()
plt.savefig("均匀分布样本_1000.png")
plt.show()

# 绘制样本的分布图
plt.figure(figsize=(12, 5))
plt.plot(uniform_samples, "o-", alpha=0.7, label="1000个样本")
plt.title("均匀分布样本分布 (1000个样本)")
plt.xlabel("索引")
plt.ylabel("样本值")
plt.legend()
plt.savefig("均匀分布样本分布_1000.png")
plt.show()

# （5）比较正态分布和均匀分布的结果
gaussian_samples = np.random.normal(mean, std_dev, 1000)
normal_skewness = skew(gaussian_samples)
normal_kurtosis = kurtosis(gaussian_samples)

uniform_skewness = skew(uniform_samples)
uniform_kurtosis = kurtosis(uniform_samples)

print(f"正态分布样本偏度: {normal_skewness:.4f}, 峰度: {normal_kurtosis:.4f}")
print(f"均匀分布样本偏度: {uniform_skewness:.4f}, 峰度: {uniform_kurtosis:.4f}")

if normal_skewness != uniform_skewness or normal_kurtosis != uniform_kurtosis:
    print("两个分布的三阶中心距和四阶中心距不相同。")
else:
    print("两个分布的三阶中心距和四阶中心距相同。")
