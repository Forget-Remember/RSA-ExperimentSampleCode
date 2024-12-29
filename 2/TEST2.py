import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 任务（1）：产生均值、协方差矩阵如下的样本数据3000个
mu_1 = np.array([0, 0])
sigma_1 = np.array([[1, 0], [0, 1]])
samples_1 = np.random.multivariate_normal(mu_1, sigma_1, 3000)

# 计算概率密度
x, y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
pos = np.dstack((x, y))
y_pdf_1 = multivariate_normal.pdf(pos, mean=mu_1, cov=sigma_1)

# 绘制三维概率密度图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples_1[:, 0], samples_1[:, 1], multivariate_normal.pdf(samples_1, mean=mu_1, cov=sigma_1), c='r', marker='o', s=5)
ax.set_title('三维概率密度图 (均值=[0,0], 协方差=[[1,0],[0,1]])')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('概率密度')
plt.savefig("三维概率密度图_均值0_0_协方差1_0_0_1.png")
plt.show()

# 绘制俯视图（空间分布特性）
plt.figure(figsize=(12, 5))
plt.scatter(samples_1[:, 0], samples_1[:, 1], alpha=0.5)
plt.title('俯视图 (均值=[0,0], 协方差=[[1,0],[0,1]])')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid(True)
plt.savefig("俯视图_均值0_0_协方差1_0_0_1.png")
plt.show()

# 绘制侧视图（边缘分布）
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].hist(samples_1[:, 0], bins=30, density=True, alpha=0.7, color='blue')
axs[0].set_title('X轴边缘分布')
axs[0].set_xlabel('X轴')
axs[0].set_ylabel('概率密度')

axs[1].hist(samples_1[:, 1], bins=30, density=True, alpha=0.7, color='green')
axs[1].set_title('Y轴边缘分布')
axs[1].set_xlabel('Y轴')
axs[1].set_ylabel('概率密度')

plt.tight_layout()
plt.savefig("侧视图_均值0_0_协方差1_0_0_1.png")
plt.show()

# 任务（2）：求取样本数据的统计量并与理论值比较
mean_estimated_1 = np.mean(samples_1, axis=0)
cov_estimated_1 = np.cov(samples_1.T)

print(f"Estimated Mean: {mean_estimated_1}")
print(f"Theoretical Mean: {mu_1}\n")

print(f"Estimated Covariance Matrix:\n{cov_estimated_1}")
print(f"Theoretical Covariance Matrix:\n{sigma_1}")

# 任务（3）：产生相关系数不同的样本数据
rho = 0.8
sigma_2 = np.array([[1, rho], [rho, 1]])
samples_2 = np.random.multivariate_normal(mu_1, sigma_2, 3000)

# 计算概率密度
y_pdf_2 = multivariate_normal.pdf(pos, mean=mu_1, cov=sigma_2)

# 绘制三维概率密度图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples_2[:, 0], samples_2[:, 1], multivariate_normal.pdf(samples_2, mean=mu_1, cov=sigma_2), c='b', marker='o', s=5)
ax.set_title('三维概率密度图 (均值=[0,0], 协方差=[[1,0.8],[0.8,1]])')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('概率密度')
plt.savefig("三维概率密度图_均值0_0_协方差1_0.8_0.8_1.png")
plt.show()

# 比较两个PDF
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, y_pdf_1, cmap='Reds', alpha=0.6, label='Cov=[[1,0],[0,1]]')
ax.plot_surface(x, y, y_pdf_2, cmap='Blues', alpha=0.6, label='Cov=[[1,0.8],[0.8,1]]')
ax.set_title('PDF对比 (不同相关系数)')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('概率密度')
ax.legend()
plt.savefig("PDF对比_不同相关系数.png")
plt.show()



