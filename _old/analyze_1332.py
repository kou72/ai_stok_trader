import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（文字化け防止）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== データ読み込み =====
csv_path = './csv_20260126_003947/1332.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

print("===== データの先頭5行 =====")
print(df.head())

print("\n===== データの基本情報 =====")
df.info()

print("\n===== 統計情報 =====")
print(df.describe())

# ===== データ前処理 =====
# 日付を日付型に変換
df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

# 数値列を数値型に変換
numeric_cols = ['始値', '高値', '安値', '終値', '出来高', '上昇率', '基準超']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 欠損値を削除
df = df.dropna()

# 日付順にソート
df = df.sort_values('日付')

print(f"\n===== データ件数: {len(df)} 日分 =====")
print(f"期間: {df['日付'].min()} ～ {df['日付'].max()}")
print(f"基準超=1の件数: {df['基準超'].sum()} 件 ({df['基準超'].mean():.2%})")

# ===== 詳細分析 =====
print("\n===== 詳細分析 =====")

# 上昇率の分布
print(f"\n上昇率の統計:")
print(f"  平均: {df['上昇率'].mean():.4f}")
print(f"  中央値: {df['上昇率'].median():.4f}")
print(f"  最大: {df['上昇率'].max():.4f}")
print(f"  最小: {df['上昇率'].min():.4f}")

# 基準超の分析
print(f"\n基準超の分析:")
print(f"  総日数: {len(df)} 日")
print(f"  基準超=1: {df['基準超'].sum()} 日 ({df['基準超'].mean():.2%})")
print(f"  基準超=0: {len(df) - df['基準超'].sum()} 日 ({1-df['基準超'].mean():.2%})")

# 最近の基準超発生日
recent_base_over = df[df['基準超'] == 1].tail(10)
if len(recent_base_over) > 0:
    print(f"\n最近の基準超発生日（最新10件）:")
    print(recent_base_over[['日付', '始値', '終値', '上昇率']])

# ===== グラフ表示 =====
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# グラフ1: 株価（4本値）
axes[0].plot(df['日付'], df['始値'], label='始値', alpha=0.7)
axes[0].plot(df['日付'], df['高値'], label='高値', alpha=0.7)
axes[0].plot(df['日付'], df['安値'], label='安値', alpha=0.7)
axes[0].plot(df['日付'], df['終値'], label='終値', linewidth=2)
axes[0].set_title('株価推移（1332）', fontsize=14, fontweight='bold')
axes[0].set_ylabel('価格（円）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# グラフ2: 出来高
axes[1].bar(df['日付'], df['出来高'], alpha=0.6, color='steelblue')
axes[1].set_title('出来高', fontsize=14, fontweight='bold')
axes[1].set_ylabel('出来高')
axes[1].grid(True, alpha=0.3)

# グラフ3: 上昇率
axes[2].plot(df['日付'], df['上昇率'], color='green', alpha=0.7)
axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='基準線(1.0)')
axes[2].axhline(y=1.03, color='red', linestyle='--', alpha=0.5, label='基準超(1.03)')
axes[2].set_title('上昇率（終値/始値）', fontsize=14, fontweight='bold')
axes[2].set_ylabel('上昇率')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# グラフ4: 基準超
base_over_dates = df[df['基準超'] == 1]['日付']
base_over_prices = df[df['基準超'] == 1]['終値']

axes[3].plot(df['日付'], df['終値'], color='blue', alpha=0.5, label='終値')
axes[3].scatter(base_over_dates, base_over_prices, color='red', s=50, alpha=0.8, label='基準超=1', zorder=5)
axes[3].set_title('基準超の発生（3%以上上昇した日）', fontsize=14, fontweight='bold')
axes[3].set_ylabel('終値（円）')
axes[3].set_xlabel('日付')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== 分布グラフ =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 上昇率のヒストグラム
axes[0].hist(df['上昇率'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(x=1.0, color='gray', linestyle='--', label='1.0')
axes[0].axvline(x=1.03, color='red', linestyle='--', label='1.03（基準）')
axes[0].set_title('上昇率の分布', fontsize=12, fontweight='bold')
axes[0].set_xlabel('上昇率')
axes[0].set_ylabel('頻度')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 基準超の円グラフ
base_counts = df['基準超'].value_counts()
colors = ['lightcoral', 'lightblue']
labels = [f'基準超=1\n({base_counts.get(1, 0)}日)', f'基準超=0\n({base_counts.get(0, 0)}日)']
axes[1].pie([base_counts.get(1, 0), base_counts.get(0, 0)], 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)
axes[1].set_title('基準超の割合', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()