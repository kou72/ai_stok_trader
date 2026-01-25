import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. CSVデータ読み込み =====
print("=" * 60)
print("1. CSVデータ読み込み")
print("=" * 60)

csv_path = './csv_20260126_003947/1332.csv'
df_original = pd.read_csv(csv_path, encoding='utf-8')

print(f"元データ件数: {len(df_original)}")
print(f"\n元データの先頭5行:")
print(df_original.head())

# ===== 2. データ前処理 =====
print("\n" + "=" * 60)
print("2. データ前処理・正規化")
print("=" * 60)

df = df_original.copy()

# 2-1. 日付を日付型に変換
df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

# 2-2. 数値列を数値型に変換
numeric_cols = ['始値', '高値', '安値', '終値', '出来高', '上昇率', '基準超']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 欠損値の確認
print(f"\n欠損値の数:")
print(df.isnull().sum())

# 2-3. 欠損値を削除
df = df.dropna()
print(f"\n欠損値削除後のデータ件数: {len(df)}")

# 2-4. 異常値除去
df = df[df['出来高'] > 0]
df = df[(df['始値'] > 0) & (df['終値'] > 0)]
print(f"異常値削除後のデータ件数: {len(df)}")

# 2-5. 日付順にソート
df = df.sort_values('日付').reset_index(drop=True)

# 2-6. 特徴量エンジニアリング
print("\n特徴量エンジニアリング中...")

# 移動平均（5日、20日、60日）
df['MA5'] = df['終値'].rolling(window=5).mean()
df['MA20'] = df['終値'].rolling(window=20).mean()
df['MA60'] = df['終値'].rolling(window=60).mean()

# ボラティリティ（20日）
df['volatility_20'] = df['上昇率'].rolling(window=20).std()

# RSI（14日）
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['終値'])

# 日中変動率
df['日中変動率'] = (df['高値'] - df['安値']) / df['始値']

# 欠損値削除（移動平均などで発生）
df_before_fillna = df.copy()
df = df.dropna()
print(f"特徴量追加後の欠損値削除: {len(df_before_fillna)} → {len(df)}")

# 2-7. 正規化
print("\n正規化中...")

feature_cols = [
    '始値', '高値', '安値', '終値', '出来高',
    '上昇率', 'MA5', 'MA20', 'MA60',
    'volatility_20', 'RSI', '日中変動率'
]

# 正規化前のデータを保存
df_before_normalize = df.copy()

# StandardScalerで正規化
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print(f"\n正規化完了！")
print(f"最終データ件数: {len(df)}")
print(f"期間: {df['日付'].min()} ～ {df['日付'].max()}")

# ===== 3. 処理後データの統計情報 =====
print("\n" + "=" * 60)
print("3. 処理後データの統計情報")
print("=" * 60)

print("\n【正規化前】")
print(df_before_normalize[feature_cols].describe())

print("\n【正規化後】")
print(df[feature_cols].describe())

# ===== 4. 可視化 =====
print("\n" + "=" * 60)
print("4. データ可視化")
print("=" * 60)

# グラフ1: 正規化前のデータ（元データ）
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('【正規化前】元データの可視化', fontsize=16, fontweight='bold')

# 株価推移
axes[0, 0].plot(df_before_normalize['日付'], df_before_normalize['終値'], label='終値', linewidth=2)
axes[0, 0].plot(df_before_normalize['日付'], df_before_normalize['MA5'], label='MA5', alpha=0.7)
axes[0, 0].plot(df_before_normalize['日付'], df_before_normalize['MA20'], label='MA20', alpha=0.7)
axes[0, 0].plot(df_before_normalize['日付'], df_before_normalize['MA60'], label='MA60', alpha=0.7)
axes[0, 0].set_title('株価と移動平均')
axes[0, 0].set_ylabel('価格（円）')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 出来高
axes[0, 1].bar(df_before_normalize['日付'], df_before_normalize['出来高'], alpha=0.6, color='steelblue')
axes[0, 1].set_title('出来高')
axes[0, 1].set_ylabel('出来高')
axes[0, 1].grid(True, alpha=0.3)

# ボラティリティ
axes[1, 0].plot(df_before_normalize['日付'], df_before_normalize['volatility_20'], color='red', alpha=0.7)
axes[1, 0].set_title('ボラティリティ（20日）')
axes[1, 0].set_ylabel('標準偏差')
axes[1, 0].grid(True, alpha=0.3)

# RSI
axes[1, 1].plot(df_before_normalize['日付'], df_before_normalize['RSI'], color='purple', alpha=0.7)
axes[1, 1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='買われすぎ(70)')
axes[1, 1].axhline(y=30, color='blue', linestyle='--', alpha=0.5, label='売られすぎ(30)')
axes[1, 1].set_title('RSI（14日）')
axes[1, 1].set_ylabel('RSI')
axes[1, 1].set_ylim(0, 100)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 上昇率
axes[2, 0].plot(df_before_normalize['日付'], df_before_normalize['上昇率'], color='green', alpha=0.7)
axes[2, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[2, 0].axhline(y=1.03, color='red', linestyle='--', alpha=0.5, label='基準超(1.03)')
axes[2, 0].set_title('上昇率（終値/始値）')
axes[2, 0].set_ylabel('上昇率')
axes[2, 0].set_xlabel('日付')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 日中変動率
axes[2, 1].plot(df_before_normalize['日付'], df_before_normalize['日中変動率'], color='orange', alpha=0.7)
axes[2, 1].set_title('日中変動率')
axes[2, 1].set_ylabel('(高値-安値)/始値')
axes[2, 1].set_xlabel('日付')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# グラフ2: 正規化後のデータ
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('【正規化後】標準化されたデータ', fontsize=16, fontweight='bold')

# 終値（正規化後）
axes[0, 0].plot(df['日付'], df['終値'], label='終値（正規化）', linewidth=2, color='blue')
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='平均(0)')
axes[0, 0].set_title('正規化後の終値')
axes[0, 0].set_ylabel('標準化された値')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 出来高（正規化後）
axes[0, 1].plot(df['日付'], df['出来高'], color='steelblue', alpha=0.7)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].set_title('正規化後の出来高')
axes[0, 1].set_ylabel('標準化された値')
axes[0, 1].grid(True, alpha=0.3)

# 移動平均（正規化後）
axes[1, 0].plot(df['日付'], df['MA5'], label='MA5', alpha=0.7)
axes[1, 0].plot(df['日付'], df['MA20'], label='MA20', alpha=0.7)
axes[1, 0].plot(df['日付'], df['MA60'], label='MA60', alpha=0.7)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_title('正規化後の移動平均')
axes[1, 0].set_ylabel('標準化された値')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# RSI（正規化後）
axes[1, 1].plot(df['日付'], df['RSI'], color='purple', alpha=0.7)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_title('正規化後のRSI')
axes[1, 1].set_ylabel('標準化された値')
axes[1, 1].grid(True, alpha=0.3)

# ボラティリティ（正規化後）
axes[2, 0].plot(df['日付'], df['volatility_20'], color='red', alpha=0.7)
axes[2, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2, 0].set_title('正規化後のボラティリティ')
axes[2, 0].set_ylabel('標準化された値')
axes[2, 0].set_xlabel('日付')
axes[2, 0].grid(True, alpha=0.3)

# 全特徴量の分布（ヒストグラム）
for col in ['終値', '出来高', 'MA20', 'RSI', 'volatility_20']:
    axes[2, 1].hist(df[col], bins=30, alpha=0.5, label=col)
axes[2, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2, 1].set_title('正規化後の特徴量分布')
axes[2, 1].set_xlabel('標準化された値')
axes[2, 1].set_ylabel('頻度')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# グラフ3: 正規化前後の比較（ヒートマップ風）
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
fig.suptitle('正規化前後の比較', fontsize=16, fontweight='bold')

# 正規化前
axes[0].plot(df_before_normalize['日付'], df_before_normalize['終値'], label='終値')
axes[0].plot(df_before_normalize['日付'], df_before_normalize['MA20'], label='MA20')
axes[0].plot(df_before_normalize['日付'], df_before_normalize['RSI'] * 10, label='RSI×10')
axes[0].set_title('正規化前（スケールが異なる）')
axes[0].set_ylabel('値')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 正規化後
axes[1].plot(df['日付'], df['終値'], label='終値')
axes[1].plot(df['日付'], df['MA20'], label='MA20')
axes[1].plot(df['日付'], df['RSI'], label='RSI')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('正規化後（同じスケールで比較可能）')
axes[1].set_ylabel('標準化された値')
axes[1].set_xlabel('日付')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)
print(f"処理後データ保存先: df（変数に格納済み）")
print(f"使用可能な特徴量: {feature_cols}")