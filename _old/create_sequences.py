import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("LSTM用シーケンスデータ作成")
print("=" * 80)

# ===== 設定 =====
CSV_PATH = './csv_20260126_003947/1332.csv'
TIME_STEP = 480  # 過去480日分（約2年）

# ===== 1. CSVデータ読み込み =====
print("\n" + "=" * 80)
print("1. CSVデータ読み込み")
print("=" * 80)

df_original = pd.read_csv(CSV_PATH, encoding='utf-8')

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
print(df.isnull().sum())

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

# 統計情報の表示
print("\n【正規化前の統計】")
print(df[feature_cols].describe())

print("\n【正規化後の統計】")
print(df_normalized[feature_cols].describe())

# ===== 3. time_step=480でシーケンス作成 =====
print("\n" + "=" * 80)
print("3. time_step=480でシーケンス作成")
print("=" * 80)

def create_sequences(data, feature_cols, target_col, time_step=480):
    """
    時系列データからシーケンスを作成
    
    Parameters:
    -----------
    data : DataFrame
        正規化済みのデータ
    feature_cols : list
        特徴量の列名リスト
    target_col : str
        目的変数の列名（'基準超'）
    time_step : int
        過去何日分のデータを使うか
    
    Returns:
    --------
    X : ndarray
        特徴量のシーケンス (サンプル数, time_step, 特徴量数)
    y : ndarray
        目的変数 (サンプル数,)
    dates : list
        各サンプルの日付
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

if X is not None:
    print(f"\n✅ シーケンス作成完了！")
    print(f"\nシーケンスデータの形状:")
    print(f"  X.shape: {X.shape}")
    print(f"    → (サンプル数={X.shape[0]}, 過去の日数={X.shape[1]}, 特徴量数={X.shape[2]})")
    print(f"  y.shape: {y.shape}")
    print(f"    → (サンプル数={y.shape[0]},)")
    
    print(f"\n目的変数（基準超）の分布:")
    print(f"  基準超=0: {np.sum(y==0)} 件 ({np.sum(y==0)/len(y)*100:.2f}%)")
    print(f"  基準超=1: {np.sum(y==1)} 件 ({np.sum(y==1)/len(y)*100:.2f}%)")
    
    # ===== 4. データの可視化 =====
    print("\n" + "=" * 80)
    print("4. データの可視化")
    print("=" * 80)
    
    # 1つのサンプルを確認
    print(f"\n1つ目のサンプル:")
    print(f"  期間: {df_normalized['日付'].iloc[0]} ～ {dates[0]}")
    print(f"  特徴量の形状: {X[0].shape}")
    print(f"  目的変数（基準超）: {y[0]}")
    
    # シーケンスの最初の数日分を表示
    print(f"\n最初の5日分の正規化データ（1つ目のサンプル）:")
    sample_df = pd.DataFrame(X[0][:5], columns=feature_cols)
    print(sample_df)
    
    # グラフ: シーケンスの可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle(f'1つ目のシーケンス（480日分）の可視化\n予測対象日: {dates[0]}, 基準超: {y[0]}', 
                 fontsize=14, fontweight='bold')
    
    # 時間軸（0日目～479日目）
    time_axis = np.arange(TIME_STEP)
    
    # グラフ1: 株価関連（正規化済み）
    axes[0].plot(time_axis, X[0][:, feature_cols.index('終値')], label='終値', linewidth=2)
    axes[0].plot(time_axis, X[0][:, feature_cols.index('MA5')], label='MA5', alpha=0.7)
    axes[0].plot(time_axis, X[0][:, feature_cols.index('MA20')], label='MA20', alpha=0.7)
    axes[0].plot(time_axis, X[0][:, feature_cols.index('MA60')], label='MA60', alpha=0.7)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('株価と移動平均（正規化済み）')
    axes[0].set_ylabel('標準化された値')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # グラフ2: テクニカル指標
    axes[1].plot(time_axis, X[0][:, feature_cols.index('RSI')], label='RSI', color='purple')
    axes[1].plot(time_axis, X[0][:, feature_cols.index('volatility_20')], label='ボラティリティ', color='red')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('テクニカル指標（正規化済み）')
    axes[1].set_ylabel('標準化された値')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # グラフ3: 出来高
    axes[2].plot(time_axis, X[0][:, feature_cols.index('出来高')], label='出来高', color='steelblue')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('出来高（正規化済み）')
    axes[2].set_ylabel('標準化された値')
    axes[2].set_xlabel('過去の日数（0日前～479日前）')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # グラフ: 全サンプルの基準超分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 円グラフ
    colors = ['lightblue', 'lightcoral']
    labels = [f'基準超=0\n({np.sum(y==0)}個)', f'基準超=1\n({np.sum(y==1)}個)']
    axes[0].pie([np.sum(y==0), np.sum(y==1)], 
                labels=labels, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90)
    axes[0].set_title('目的変数の分布')
    
    # 時系列での基準超の発生
    axes[1].scatter(range(len(y)), y, alpha=0.3, s=10)
    axes[1].set_title('時系列での基準超の発生')
    axes[1].set_xlabel('サンプル番号（時系列順）')
    axes[1].set_ylabel('基準超（0 or 1）')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print(f"\n次のステップ:")
    print(f"  - 訓練/検証/テストに分割")
    print(f"  - LSTMモデルの構築")
    print(f"  - 訓練と評価")

else:
    print("\nエラー: シーケンス作成に失敗しました")