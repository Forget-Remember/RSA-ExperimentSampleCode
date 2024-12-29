import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 步骤（1）：设计带通 FIR 滤波器
N = 4096  # 滤波器阶数
lowcut = 0.15
highcut = 0.35
nyquist = 0.5  # Nyquist 频率

# 设计带通滤波器
taps = firwin(N, [lowcut, highcut], pass_zero=False, nyq=nyquist)

# 计算频率响应
w, h = freqz(taps, worN=8000)
freqs = w / (2 * np.pi)  # 数字角频率归一化到 [0, 1]

# 绘制幅频特性曲线
plt.figure(figsize=(12, 5))
plt.plot(freqs, 20 * np.log10(abs(h)), label="幅度响应")
plt.title("带通 FIR 滤波器的幅频特性")
plt.xlabel("数字角频率")
plt.ylabel("幅度 (dB)")
plt.axvline(lowcut, color="red", linestyle="--", label=f"Lowcut={lowcut}")
plt.axvline(highcut, color="red", linestyle="--", label=f"Highcut={highcut}")
plt.legend()
plt.grid(True)
plt.savefig("幅频特性.png")
plt.show()

# 绘制相频特性曲线
plt.figure(figsize=(12, 5))
plt.plot(freqs, np.angle(h), label="相位响应")
plt.title("带通 FIR 滤波器的相频特性")
plt.xlabel("数字角频率")
plt.ylabel("相位 (radians)")
plt.axvline(lowcut, color="red", linestyle="--", label=f"Lowcut={lowcut}")
plt.axvline(highcut, color="red", linestyle="--", label=f"Highcut={highcut}")
plt.legend()
plt.grid(True)
plt.savefig("相频特性.png")
plt.show()

# 步骤（2）：将实验五第三节实验内容及要求中步骤（1）中所产生的序列作为滤波器输入
# 生成高斯序列
mean = 0
std_dev = 1
samples = np.random.normal(mean, std_dev, N)

# 应用滤波器
filtered_samples = lfilter(taps, 1.0, samples)

# 绘制滤波器输入输出的时域波形
plt.figure(figsize=(12, 5))
plt.plot(samples[:200], label="输入信号")
plt.plot(filtered_samples[:200], label="输出信号")
plt.title("滤波器输入输出的时域波形")
plt.xlabel("样本索引")
plt.ylabel("幅度")
plt.legend()
plt.grid(True)
plt.savefig("时域波形.png")
plt.show()

# 计算输出过程的均值及方差
output_mean = np.mean(filtered_samples)
output_var = np.var(filtered_samples)

print(f"输出信号的均值: {output_mean}")
print(f"输出信号的方差: {output_var}")

# 步骤（3）：用分段法估计输入信号和输出信号的频谱
M = 512
num_segments_input = len(samples) // M
num_segments_output = len(filtered_samples) // M

psd_segments_input = []
psd_segments_output = []

for i in range(num_segments_input):
    segment_input = samples[i * M : (i + 1) * M]
    fft_segment_input = np.fft.fft(segment_input)
    psd_segment_input = (np.abs(fft_segment_input) ** 2) / M
    psd_segments_input.append(psd_segment_input)

for i in range(num_segments_output):
    segment_output = filtered_samples[i * M : (i + 1) * M]
    fft_segment_output = np.fft.fft(segment_output)
    psd_segment_output = (np.abs(fft_segment_output) ** 2) / M
    psd_segments_output.append(psd_segment_output)

psd_avg_input = np.mean(psd_segments_input, axis=0)
psd_avg_output = np.mean(psd_segments_output, axis=0)
freqs_psd = np.fft.fftfreq(M) * 2 * np.pi  # 数字角频率

# 绘制输入输出的信号的功率谱密度
plt.figure(figsize=(12, 5))
plt.plot(freqs_psd[: M // 2], 10 * np.log10(psd_avg_input[: M // 2]), label="输入信号")
plt.plot(freqs_psd[: M // 2], 10 * np.log10(psd_avg_output[: M // 2]), label="输出信号")
plt.title("输入输出信号的功率谱密度")
plt.xlabel("数字角频率 (rad/sample)")
plt.ylabel("功率谱密度 (dB)")
plt.legend()
plt.grid(True)
plt.savefig("功率谱密度对比.png")
plt.show()
