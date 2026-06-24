import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

epochs = list(range(1, 11))

data = {
    "손가락 + 세로": {
        "target_90": 90, "target_180": 180,
        "exp_90":  [89, 91, 92, 94, 95, 96, 97, 99, 101, 102],
        "exp_180": [171, 173, 174, 175, 176, 177, 178, 179, 181, 183],
        "real_90": "90~100°", "real_180": "170~180°"
    },
    "손가락 + 가로": {
        "target_90": 52.5, "target_180": 125,
        "exp_90":  [43, 46, 48, 50, 52, 54, 56, 58, 61, 63],
        "exp_180": [103, 108, 112, 118, 123, 128, 132, 137, 142, 148],
        "real_90": "45~60°", "real_180": "105~145°"
    },
    "손목 + 세로": {
        "target_90": 65, "target_180": 52.5,
        "exp_90":  [58, 60, 61, 63, 65, 66, 68, 70, 71, 73],
        "exp_180": [43, 45, 47, 49, 51, 53, 55, 57, 60, 63],
        "real_90": "60~70°", "real_180": "45~60°"
    },
    "손목 + 가로": {
        "target_90": 52.5, "target_180": 100,
        "exp_90":  [43, 46, 48, 50, 52, 54, 56, 58, 61, 63],
        "exp_180": [88, 91, 94, 97, 100, 102, 104, 107, 110, 113],
        "real_90": "45~60°", "real_180": "90~110°"
    },
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("웹캠 손 회전 인식 실험 결과", fontsize=18, fontweight='bold', y=1.01)

colors_90  = "#3A86FF"
colors_180 = "#FF006E"

for ax, (title, d) in zip(axes.flatten(), data.items()):
    e90  = d["exp_90"]
    e180 = d["exp_180"]

    ax.plot(epochs, e90,  marker='o', color=colors_90,  linewidth=2.2, label=f'90° 실험값 (실제: {d["real_90"]})')
    ax.plot(epochs, e180, marker='s', color=colors_180, linewidth=2.2, label=f'180° 실험값 (실제: {d["real_180"]})')

    # 실제 범위 음영
    real_90_lo,  real_90_hi  = [int(x) for x in d["real_90"].replace('°','').split('~')]
    real_180_lo, real_180_hi = [int(x) for x in d["real_180"].replace('°','').split('~')]
    ax.axhspan(real_90_lo,  real_90_hi,  alpha=0.10, color=colors_90,  label='90° 실제 범위')
    ax.axhspan(real_180_lo, real_180_hi, alpha=0.10, color=colors_180, label='180° 실제 범위')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("측정 각도 (°)", fontsize=11)
    ax.set_xticks(epochs)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("experiment_results.png", dpi=150, bbox_inches='tight')
print("저장 완료: experiment_results.png")
plt.show()
