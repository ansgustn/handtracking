import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

final_df = pd.read_csv("multicam_evaluation_snapshot.csv")

target_angles = [90, 180, 270, 360]
stages = ["MP", "RTM", "Frei"]
cam_names = ["Camera_1", "Camera_2", "Camera_3"]

plt.figure(figsize=(12, 7))
x = np.arange(len(target_angles))
width = 0.25

bar_colors = {"MP": "#1f77b4", "RTM": "#ff7f0e", "Frei": "#2ca02c"}

multiplier = 0
for model in stages:
    bars = []
    for target in target_angles:
        # Filter for current model and target
        row = final_df[(final_df['Model'] == model) & (final_df['TargetAngle'] == target)]
        if not row.empty:
            cam_vals = []
            for cam in cam_names:
                val = row.iloc[0][cam]
                if pd.notna(val):
                    cam_vals.append(abs(val))
            
            if cam_vals:
                bars.append(np.max(cam_vals))
            else:
                bars.append(0)
        else:
            bars.append(0)
            
    offset = width * multiplier
    rects = plt.bar(x + offset, bars, width, 
                    label=model, 
                    color=bar_colors.get(model),
                    edgecolor='white', alpha=0.9)
    
    labels = [f"{v:.1f}°" if v != 0 else "-" for v in bars]
    plt.bar_label(rects, labels=labels, padding=3, fontsize=10)
    
    multiplier += 1

plt.title("Model Evaluation Snapshot (Maximum of All Cameras)", fontsize=16, fontweight='bold')
plt.xlabel("Target Angle")
plt.ylabel("Maximum Measured Angle (Degree) [Absolute]")
plt.xticks(x + width, [f"{t}°" for t in target_angles])

plt.legend(loc='upper right', title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

ax = plt.gca()
ax.set_ylim(0, max(400, ax.get_ylim()[1] * 1.1))

plt.tight_layout()
plt.savefig("multicam_evaluation_snapshot_bar.png", dpi=300)
print("최댓값 통합 그래프 재생성 완료!")
