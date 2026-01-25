import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("LSTM用データ準備（訓練/検証/テスト分割まで）")
print("=" * 80)

# ===== 設定 =====
CSV_PATH = './csv_20260126_003947/1332.csv'
TIME_STEP = 480  # 過去480日分（約2年）
TRAIN_RATIO = 0.7   # 訓練データ 70%
VAL_RATIO = 0.15    # 検証データ 15%
TEST_RATIO = 0.15   # テストデータ 15%

# ===== 1. CSVデータ読み込み =====
print("\n" + "=" * 80)
print("1. CSVデータ読み込み")
print("=" * 80)

df_original = pd.read_csv(CSV_PATH, encoding='utf-8')

print(f"ファイルパス: {CSV_PATH}")
print(f"元データ件数: {len(df_original)}")
print(f"\n元データの先頭5行:")
print(df_original.head())
print(f"\n元データの列:")
print(df_original.columns.tolist())

# ===== 2. データ前処理・正規化 =====
print("\n" + "=" * 80)
print("2. データ前処理・正規化")
print("=" * 80)

df = df_original.copy()

# 2-1. 日付を日付型に変換
print("\n[2-1] 日付型に変換中...")
df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

# 2-2. 数値列を数値型に変換
print("[2-2] 数値型に変換中...")
numeric_cols = ['始値', '高値', '安値', '終値', '出来高', '上昇率', '基準超']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 欠損値の確認
print(f"\n欠損値の数:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "欠損値なし")

# 2-3. 欠損値を削除
df = df.dropna()
print(f"\n欠損値削除後: {len(df)} 件")

# 2-4. 異常値除去
print("\n[2-4] 異常値除去中...")
df = df[df['出来高'] > 0]
df = df[(df['始値'] > 0) & (df['終値'] > 0)]
print(f"異常値削除後: {len(df)} 件")

# 2-5. 日付順にソート
print("\n[2-5] 日付順にソート中...")
df = df.sort_values('日付').reset_index(drop=True)
print(f"期間: {df['日付'].min()} ～ {df['日付'].max()}")

# 2-6. 特徴量エンジニアリング
print("\n[2-6] 特徴量エンジニアリング中...")

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

# 前日比
df['前日比'] = df['終値'].pct_change()

# 欠損値削除（移動平均などで発生）
df = df.dropna()
print(f"特徴量追加後: {len(df)} 件")

# 2-7. 正規化
print("\n[2-7] 正規化中...")

feature_cols = [
    '始値', '高値', '安値', '終値', '出来高',
    '上昇率', 'MA5', 'MA20', 'MA60',
    'volatility_20', 'RSI', '日中変動率', '前日比'
]

print(f"使用する特徴量（{len(feature_cols)}個）:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# 正規化前のデータを保存（可視化用）
df_before_normalize = df.copy()

# StandardScalerで正規化
scaler = StandardScaler()
df_normalized = df.copy()
df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

print(f"\n正規化完了！")
print(f"最終データ件数: {len(df_normalized)}")

# ===== 3. time_step=480でシーケンス作成 =====
print("\n" + "=" * 80)
print("3. time_step=480でシーケンス作成")
print("=" * 80)

def create_sequences(data, feature_cols, target_col, time_step=480):
    """
    時系列データからシーケンスを作成
    """
    X, y, dates = [], [], []
    
    # データが十分にあるか確認
    if len(data) <= time_step:
        print(f"エラー: データ件数({len(data)})がtime_step({time_step})以下です")
        return None, None, None
    
    # シーケンスを作成
    for i in range(len(data) - time_step):
        # 過去time_step日分の特徴量
        X.append(data[feature_cols].iloc[i:i+time_step].values)
        
        # time_step日後の基準超
        y.append(data[target_col].iloc[i+time_step])
        
        # 記録用の日付
        dates.append(data['日付'].iloc[i+time_step])
    
    return np.array(X), np.array(y), dates

print(f"\ntime_step: {TIME_STEP} 日")
print(f"データ件数: {len(df_normalized)} 件")
print(f"作成可能なシーケンス数: {len(df_normalized) - TIME_STEP} 個")

# シーケンス作成
X, y, dates = create_sequences(
    df_normalized, 
    feature_cols, 
    '基準超', 
    TIME_STEP
)

if X is None:
    print("エラー: シーケンス作成に失敗しました")
    exit()

print(f"\n✅ シーケンス作成完了！")
print(f"\nシーケンスデータの形状:")
print(f"  X.shape: {X.shape}")
print(f"    → (サンプル数={X.shape[0]}, 過去の日数={X.shape[1]}, 特徴量数={X.shape[2]})")
print(f"  y.shape: {y.shape}")
print(f"    → (サンプル数={y.shape[0]},)")

print(f"\n目的変数（基準超）の分布:")
print(f"  基準超=0: {np.sum(y==0)} 件 ({np.sum(y==0)/len(y)*100:.2f}%)")
print(f"  基準超=1: {np.sum(y==1)} 件 ({np.sum(y==1)/len(y)*100:.2f}%)")

# ===== 4. 訓練/検証/テスト分割 =====
print("\n" + "=" * 80)
print("4. 訓練/検証/テスト分割")
print("=" * 80)

# 時系列データなので、時間順に分割（ランダムシャッフルしない）
total_samples = len(X)
train_size = int(total_samples * TRAIN_RATIO)
val_size = int(total_samples * VAL_RATIO)

# インデックスで分割
train_end = train_size
val_end = train_size + val_size

print(f"\n分割比率:")
print(f"  訓練データ: {TRAIN_RATIO*100:.0f}%")
print(f"  検証データ: {VAL_RATIO*100:.0f}%")
print(f"  テストデータ: {TEST_RATIO*100:.0f}%")

# 訓練データ
X_train = X[:train_end]
y_train = y[:train_end]
dates_train = dates[:train_end]

# 検証データ
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
dates_val = dates[train_end:val_end]

# テストデータ
X_test = X[val_end:]
y_test = y[val_end:]
dates_test = dates[val_end:]

print(f"\n✅ 分割完了！")
print(f"\n各データセットの詳細:")
print(f"\n【訓練データ】")
print(f"  サンプル数: {len(X_train)}")
print(f"  期間: {dates_train[0]} ～ {dates_train[-1]}")
print(f"  基準超=0: {np.sum(y_train==0)} 件 ({np.sum(y_train==0)/len(y_train)*100:.2f}%)")
print(f"  基準超=1: {np.sum(y_train==1)} 件 ({np.sum(y_train==1)/len(y_train)*100:.2f}%)")

print(f"\n【検証データ】")
print(f"  サンプル数: {len(X_val)}")
print(f"  期間: {dates_val[0]} ～ {dates_val[-1]}")
print(f"  基準超=0: {np.sum(y_val==0)} 件 ({np.sum(y_val==0)/len(y_val)*100:.2f}%)")
print(f"  基準超=1: {np.sum(y_val==1)} 件 ({np.sum(y_val==1)/len(y_val)*100:.2f}%)")

print(f"\n【テストデータ】")
print(f"  サンプル数: {len(X_test)}")
print(f"  期間: {dates_test[0]} ～ {dates_test[-1]}")
print(f"  基準超=0: {np.sum(y_test==0)} 件 ({np.sum(y_test==0)/len(y_test)*100:.2f}%)")
print(f"  基準超=1: {np.sum(y_test==1)} 件 ({np.sum(y_test==1)/len(y_test)*100:.2f}%)")

# ===== 5. データの可視化 =====
print("\n" + "=" * 80)
print("5. データの可視化")
print("=" * 80)

# グラフ1: データ分割の可視化
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
fig.suptitle('訓練/検証/テストデータの分割', fontsize=16, fontweight='bold')

# 全体のタイムライン
all_dates = dates_train + dates_val + dates_test
train_indices = list(range(len(dates_train)))
val_indices = list(range(len(dates_train), len(dates_train) + len(dates_val)))
test_indices = list(range(len(dates_train) + len(dates_val), len(all_dates)))

# 上段: 分割の可視化
axes[0].barh(0, len(dates_train), left=0, color='blue', alpha=0.6, label=f'訓練 ({len(dates_train)})')
axes[0].barh(0, len(dates_val), left=len(dates_train), color='green', alpha=0.6, label=f'検証 ({len(dates_val)})')
axes[0].barh(0, len(dates_test), left=len(dates_train)+len(dates_val), color='red', alpha=0.6, label=f'テスト ({len(dates_test)})')
axes[0].set_xlim(0, len(all_dates))
axes[0].set_ylim(-0.5, 0.5)
axes[0].set_xlabel('サンプル数')
axes[0].set_yticks([])
axes[0].legend(loc='upper right')
axes[0].set_title('データセットの分割比率')
axes[0].grid(True, alpha=0.3, axis='x')

# 下段: 基準超の分布
axes[1].scatter(train_indices, y_train, alpha=0.3, s=10, color='blue', label='訓練')
axes[1].scatter(val_indices, y_val, alpha=0.3, s=10, color='green', label='検証')
axes[1].scatter(test_indices, y_test, alpha=0.3, s=10, color='red', label='テスト')
axes[1].set_xlabel('サンプル番号（時系列順）')
axes[1].set_ylabel('基準超（0 or 1）')
axes[1].set_title('時系列での基準超の分布')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# グラフ2: 各データセットの基準超の割合
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('各データセットの基準超の割合', fontsize=16, fontweight='bold')

colors = ['lightblue', 'lightcoral']

# 訓練データ
train_counts = [np.sum(y_train==0), np.sum(y_train==1)]
axes[0].pie(train_counts, 
            labels=[f'基準超=0\n({train_counts[0]})', f'基準超=1\n({train_counts[1]})'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)
axes[0].set_title(f'訓練データ\n(n={len(y_train)})')

# 検証データ
val_counts = [np.sum(y_val==0), np.sum(y_val==1)]
axes[1].pie(val_counts, 
            labels=[f'基準超=0\n({val_counts[0]})', f'基準超=1\n({val_counts[1]})'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)
axes[1].set_title(f'検証データ\n(n={len(y_val)})')

# テストデータ
test_counts = [np.sum(y_test==0), np.sum(y_test==1)]
axes[2].pie(test_counts, 
            labels=[f'基準超=0\n({test_counts[0]})', f'基準超=1\n({test_counts[1]})'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)
axes[2].set_title(f'テストデータ\n(n={len(y_test)})')

plt.tight_layout()
plt.show()

# グラフ3: データセットのサマリー
fig, ax = plt.subplots(figsize=(12, 6))

datasets = ['訓練', '検証', 'テスト']
total_counts = [len(y_train), len(y_val), len(y_test)]
zero_counts = [np.sum(y_train==0), np.sum(y_val==0), np.sum(y_test==0)]
one_counts = [np.sum(y_train==1), np.sum(y_val==1), np.sum(y_test==1)]

x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, zero_counts, width, label='基準超=0', color='lightblue')
bars2 = ax.bar(x + width/2, one_counts, width, label='基準超=1', color='lightcoral')

ax.set_xlabel('データセット')
ax.set_ylabel('サンプル数')
ax.set_title('各データセットの内訳')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 数値ラベル
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
print(f"\nデータの準備が完了しました。")
print(f"\n次のステップ:")
print(f"  1. LSTMモデルの構築")
print(f"  2. モデルの訓練")
print(f"  3. モデルの評価")
print(f"\n変数に格納されているデータ:")
print(f"  - X_train: 訓練用特徴量 {X_train.shape}")
print(f"  - y_train: 訓練用目的変数 {y_train.shape}")
print(f"  - X_val: 検証用特徴量 {X_val.shape}")
print(f"  - y_val: 検証用目的変数 {y_val.shape}")
print(f"  - X_test: テスト用特徴量 {X_test.shape}")
print(f"  - y_test: テスト用目的変数 {y_test.shape}")