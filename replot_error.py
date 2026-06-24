import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

final_df = pd.read_csv("multicam_evaluation_snapshot.csv")

target_angles = [90, 180, 270, 360]
stages = ["MP", "RTM", "Frei"]
cam_names = ["Camera_1", "Camera_2", "Camera_3"]

plt.figure(figsize=(12, 7))
x = np.arange(len(target_angles))
width = 0.25  # 막대 3개 (모델별 1개)

bar_colors = {"MP": "#1f77b4", "RTM": "#ff7f0e", "Frei": "#2ca02c"}

multiplier = 0
for model in stages:
    bars = []
    for target in target_angles:
        row = final_df[(final_df['Model'] == model) & (final_df['TargetAngle'] == target)]
        if not row.empty:
            errors = []
            for cam in cam_names:
                val = row.iloc[0][cam]
                if pd.notna(val):
                    error = abs(abs(val) - target)
                    errors.append(error)
            
            if errors:
                bars.append(np.mean(errors))
            else:
                bars.append(np.nan)
        else:
            bars.append(np.nan)
            
    offset = width * multiplier
    
    rects = plt.bar(x + offset, bars, width, 
                    label=model, 
                    color=bar_colors.get(model),
                    edgecolor='white', alpha=0.9)
    
    labels = ["Fail" if np.isnan(v) else f"{v:.1f}°" for v in bars]
    plt.bar_label(rects, labels=labels, padding=3, fontsize=10)
    
    multiplier += 1

plt.axhline(0, color='black', linewidth=2)

plt.title("Measured value", fontsize=16, fontweight='bold')
plt.xlabel("Target Angle")
plt.ylabel("AbsoluteDegree")
plt.xticks(x + width, [f"{t}°" for t in target_angles])

plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

ax = plt.gca()
ax.set_ylim(0, max(20, ax.get_ylim()[1] * 1.1))

plt.tight_layout()
plt.savefig("multicam_evaluation_snapshot_error_bar.png", dpi=300)
print("절댓값 평균 오차 그래프 (한글 폰트 적용) 재생성 완료!")
