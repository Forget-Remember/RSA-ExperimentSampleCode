import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 步骤（1）：生成高斯序列
N = 4096
mean = 0
std_dev = 1
samples = np.random.normal(mean, std_dev, N)


# 直接法估计功率谱密度
def direct_method(samples):
    N = len(samples)
    fft_result = np.fft.fft(samples)
    psd_direct = (np.abs(fft_result) ** 2) / N
    freqs = np.fft.fftfreq(N) * 2 * np.pi  # 数字角频率
    return freqs, psd_direct


freqs, psd_direct = direct_method(samples)

# 绘制直接法估计的功率谱密度
plt.figure(figsize=(12, 5))
plt.plot(freqs[: N // 2], 10 * np.log10(psd_direct[: N // 2]), label="直接法")
plt.title("直接法估计的功率谱密度")
plt.xlabel("数字角频率 (rad/sample)")
plt.ylabel("功率谱密度 (dB)")
plt.legend()
plt.grid(True)
plt.savefig("直接法功率谱密度.png")
plt.show()

# 步骤（2）：分段法估计功率谱密度
M = 512
L = 8
num_segments = N // M

psd_segments = []
for i in range(num_segments - L + 1):
    segment = samples[i * M : (i + M) * L]
    fft_segment = np.fft.fft(segment)
    psd_segment = (np.abs(fft_segment) ** 2) / (M * L)
    psd_segments.append(psd_segment)

psd_avg = np.mean(psd_segments, axis=0)
freqs_avg = np.fft.fftfreq(M * L) * 2 * np.pi  # 数字角频率

# 绘制分段法估计的功率谱密度
plt.figure(figsize=(12, 5))
plt.plot(freqs_avg[: M * L // 2], 10 * np.log10(psd_avg[: M * L // 2]), label="分段法")
plt.title("分段法估计的功率谱密度")
plt.xlabel("数字角频率 (rad/sample)")
plt.ylabel("功率谱密度 (dB)")
plt.legend()
plt.grid(True)
plt.savefig("分段法功率谱密度.png")
plt.show()

# 步骤（3）：FFT变换得到功率谱估计
fft_full = np.fft.fft(samples)
psd_fft = (np.abs(fft_full) ** 2) / N
freqs_fft = np.fft.fftfreq(N) * 2 * np.pi  # 数字角频率

# 绘制FFT变换得到的功率谱密度
plt.figure(figsize=(12, 5))
plt.plot(freqs_fft[: N // 2], 10 * np.log10(psd_fft[: N // 2]), label="FFT变换")
plt.title("FFT变换得到的功率谱密度")
plt.xlabel("数字角频率 (rad/sample)")
plt.ylabel("功率谱密度 (dB)")
plt.legend()
plt.grid(True)
plt.savefig("FFT变换功率谱密度.png")
plt.show()

# 绘制三种方法的功率谱密度对比
plt.figure(figsize=(12, 5))
plt.plot(freqs[: N // 2], 10 * np.log10(psd_direct[: N // 2]), label="直接法")
plt.plot(freqs_avg[: M * L // 2], 10 * np.log10(psd_avg[: M * L // 2]), label="分段法")
plt.plot(freqs_fft[: N // 2], 10 * np.log10(psd_fft[: N // 2]), label="FFT变换")
plt.title("三种方法的功率谱密度对比")
plt.xlabel("数字角频率 (rad/sample)")
plt.ylabel("功率谱密度 (dB)")
plt.legend()
plt.grid(True)
plt.savefig("功率谱密度对比.png")
plt.show()
