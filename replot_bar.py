import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

final_df = pd.read_csv("multicam_evaluation_angles.csv")

target_angles = [90, 180, 270, 360]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

stages = ["MP", "RTM", "Frei"]
cam_configs = [{'name': 'Camera_1'}, {'name': 'Camera_2'}, {'name': 'Camera_3'}]

bar_colors = {"MP": "#1f77b4", "RTM": "#ff7f0e", "Frei": "#2ca02c"}
ref_cam = "Camera_1"

for i, cfg in enumerate(cam_configs):
    eval_cam = cfg['name']
    ax = axes[i]
    
    x = np.arange(len(target_angles))
    width = 0.25
    
    for j, model_name in enumerate(stages):
        ref_col = f"{ref_cam}_{model_name}"
        eval_col = f"{eval_cam}_{model_name}"
        
        if ref_col not in final_df.columns or eval_col not in final_df.columns:
            continue
            
        valid_idx = final_df[ref_col].notna() & final_df[eval_col].notna()
        if not valid_idx.any():
            continue
            
        ref_angles = np.degrees(np.unwrap(np.radians(final_df.loc[valid_idx, ref_col])))
        eval_angles = np.degrees(np.unwrap(np.radians(final_df.loc[valid_idx, eval_col])))
        
        bars = []
        for target in target_angles:
            diffs = np.abs(np.abs(ref_angles) - target)
            min_diff = diffs.min()
            
            if min_diff < 15: 
                closest_idx = diffs.argmin()
                val = eval_angles.iloc[closest_idx] if hasattr(eval_angles, 'iloc') else eval_angles[closest_idx]
                bars.append(val)
            else:
                bars.append(0)
                
        offset = width * j
        rects = ax.bar(x + offset, bars, width, label=model_name, color=bar_colors.get(model_name), alpha=0.85)
        labels = [f"{v:.0f}°" if v != 0 else "-" for v in bars]
        ax.bar_label(rects, labels=labels, padding=3, fontsize=9)
        
    ax.set_title(f"{eval_cam} (Reference: {ref_cam})")
    ax.set_xlabel("Target Angle")
    if i == 0:
        ax.set_ylabel("Measured Angle (Degree)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{t}°" for t in target_angles])
    if i == 2:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')

plt.suptitle("Model Evaluation at 90°, 180°, 270°, 360° (Anchored to Camera_1)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("multicam_evaluation_bar.png", dpi=300)
print("음수 회전 대응 그래프 생성 완료!")
