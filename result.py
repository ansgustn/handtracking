import pandas as pd
import matplotlib.pyplot as plt

# 1. 한글 폰트 설정 (환경에 맞게 수정: Windows 'Malgun Gothic', Mac 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 2. 데이터 불러오기 (원본 음수/양수 유지)
df = pd.read_csv('experiment_results.csv')

# 3. 시간을 10초 기준으로 필터링
df_10s = df[df['Elapsed_Time_Sec'] <= 10.0]

# 4. 각 조건(Trial)별 양수 최댓값(max)과 음수 최댓값(min)을 동시에 추출
summary = df_10s.groupby('Trial')['Angle_Deg'].agg(['max', 'min']).reset_index()

# 조건 라벨 매핑
labels = {
    1: '세로 두 손가락 90',
    2: '가로 손 전체 90',
    3: '가로 두 손가락 90',
    4: '세로 손 전체 90',
    5: '세로 두 손가락 180',
    6: '가로 손 전체 180',
    7: '가로 두 손가락 180',
    8: '세로 손 전체 180'
}
summary['Condition'] = summary['Trial'].map(labels)

# 5. 그래프 그리기 (양방향 막대 그래프)
plt.figure(figsize=(14, 8))

# 양수 최댓값 막대 (파란색)
bars_pos = plt.bar(summary['Condition'], summary['max'], color='cornflowerblue', edgecolor='black', alpha=0.8, label='양수 최댓값')
# 음수 최댓값 막대 (빨간색)
bars_neg = plt.bar(summary['Condition'], summary['min'], color='lightcoral', edgecolor='black', alpha=0.8, label='음수 최댓값')

# 중앙 기준선 (0도) 표시
plt.axhline(0, color='black', linewidth=1.2)

# 그래프 디자인 꾸미기
plt.title('10초 이내 조건별 양수 및 음수 최댓값 비교', fontsize=16, fontweight='bold')
plt.xlabel('실험 조건', fontsize=12)
plt.ylabel('각도 (도)', fontsize=12)
plt.xticks(rotation=45, ha='right') # x축 라벨 45도 회전
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper right') # 범례 표시

# 막대 위에 구체적인 수치 텍스트 표시
# 양수 텍스트 (막대 위에 표시)
for bar in bars_pos:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2, 
             f'{height:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 음수 텍스트 (막대 아래에 표시)
for bar in bars_neg:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 5, 
             f'{height:.1f}°', ha='center', va='top', fontsize=9, fontweight='bold')

plt.tight_layout()

# 6. 고해상도 PNG 파일로 저장
plt.savefig('experiment_graph_both.png', dpi=300, bbox_inches='tight')

# 7. 화면 출력
plt.show()