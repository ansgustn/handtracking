import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

final_df = pd.read_csv("multicam_evaluation_angles.csv")

plt.figure(figsize=(15, 8))
colors = {"Camera_1": "blue", "Camera_2": "green", "Camera_3": "red"}
styles = {"MP": "-", "RTM": "--", "Frei": ":"}
stages = ["MP", "RTM", "Frei"]
cam_configs = [{'name': 'Camera_1'}, {'name': 'Camera_2'}, {'name': 'Camera_3'}]

all_angles = []

for cfg in cam_configs:
    cam_name = cfg['name']
    for model_name in stages:
        col = f"{cam_name}_{model_name}"
        time_col = f"Time_{model_name}"
        
        if col not in final_df.columns or time_col not in final_df.columns:
            continue
        if final_df[col].isna().all():
            continue
            
        valid_idx = final_df[col].notna()
        unwrapped_angles = np.degrees(np.unwrap(np.radians(final_df.loc[valid_idx, col])))
        final_df.loc[valid_idx, col] = unwrapped_angles
        all_angles.extend(unwrapped_angles)
        
        plt.plot(final_df[time_col], final_df[col], 
                 color=colors.get(cam_name, "black"), 
                 linestyle=styles[model_name],
                 linewidth=2 if model_name == "MP" else 1.5,
                 alpha=0.8,
                 label=f"{cam_name} ({model_name})")

if all_angles:
    min_ang = min(all_angles)
    max_ang = max(all_angles)
    start_tick = (int(min_ang) // 90 - 1) * 90
    end_tick = (int(max_ang) // 90 + 2) * 90
    plt.yticks(np.arange(start_tick, end_tick, 90))

plt.title("Model-Sequential Multi-Camera Z-Axis Rotation Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Angle (Degree)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6, color='gray')
plt.tight_layout()
plt.savefig("multicam_evaluation_graph.png", dpi=300)
print("새로운 그래프 생성 완료!")
