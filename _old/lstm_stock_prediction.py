import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("LSTM株価予測モデル（完全版）")
print("=" * 80)

# ===== 設定 =====
CSV_PATH = './csv_20260126_003947/1332.csv'
TIME_STEP = 480
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# LSTMモデルの設定
BATCH_SIZE = 32
EPOCHS = 10  # 時間短縮のため少なめ
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# デバイス設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用デバイス: {DEVICE}")

# ===== 1. CSVデータ読み込み =====
print("\n" + "=" * 80)
print("1. CSVデータ読み込み")
print("=" * 80)

df_original = pd.read_csv(CSV_PATH, encoding='utf-8')
print(f"元データ件数: {len(df_original)}")

# ===== 2. データ前処理・正規化 =====
print("\n" + "=" * 80)
print("2. データ前処理・正規化")
print("=" * 80)

df = df_original.copy()

# 日付を日付型に変換
df['日付'] = pd.to_datetime(df['日付'], errors='coerce')

# 数値列を数値型に変換
numeric_cols = ['始値', '高値', '安値', '終値', '出来高', '上昇率', '基準超']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 欠損値・異常値削除
df = df.dropna()
df = df[df['出来高'] > 0]
df = df[(df['始値'] > 0) & (df['終値'] > 0)]
df = df.sort_values('日付').reset_index(drop=True)

print(f"前処理後: {len(df)} 件")

# 特徴量エンジニアリング
df['MA5'] = df['終値'].rolling(window=5).mean()
df['MA20'] = df['終値'].rolling(window=20).mean()
df['MA60'] = df['終値'].rolling(window=60).mean()
df['volatility_20'] = df['上昇率'].rolling(window=20).std()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['終値'])
df['日中変動率'] = (df['高値'] - df['安値']) / df['始値']
df['前日比'] = df['終値'].pct_change()

df = df.dropna()
print(f"特徴量追加後: {len(df)} 件")

# 正規化
feature_cols = [
    '始値', '高値', '安値', '終値', '出来高',
    '上昇率', 'MA5', 'MA20', 'MA60',
    'volatility_20', 'RSI', '日中変動率', '前日比'
]

scaler = StandardScaler()
df_normalized = df.copy()
df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

print(f"正規化完了！特徴量数: {len(feature_cols)}")

# ===== 3. time_step=480でシーケンス作成 =====
print("\n" + "=" * 80)
print("3. シーケンス作成（time_step=480）")
print("=" * 80)

def create_sequences(data, feature_cols, target_col, time_step):
    X, y, dates = [], [], []
    for i in range(len(data) - time_step):
        X.append(data[feature_cols].iloc[i:i+time_step].values)
        y.append(data[target_col].iloc[i+time_step])
        dates.append(data['日付'].iloc[i+time_step])
    return np.array(X), np.array(y), dates

X, y, dates = create_sequences(df_normalized, feature_cols, '基準超', TIME_STEP)

print(f"シーケンス作成完了！")
print(f"X.shape: {X.shape} (サンプル数, 時系列長, 特徴量数)")
print(f"y.shape: {y.shape}")
print(f"基準超=1の割合: {np.sum(y==1)/len(y)*100:.2f}%")

# ===== 4. 訓練/検証/テスト分割 =====
print("\n" + "=" * 80)
print("4. 訓練/検証/テスト分割")
print("=" * 80)

total_samples = len(X)
train_end = int(total_samples * TRAIN_RATIO)
val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"訓練: {len(X_train)} サンプル")
print(f"検証: {len(X_val)} サンプル")
print(f"テスト: {len(X_test)} サンプル")

# PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== 5. LSTMモデル構築 =====
print("\n" + "=" * 80)
print("5. LSTMモデル構築")
print("=" * 80)

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 最後の時刻の出力
        last_output = lstm_out[:, -1, :]
        
        # 全結合層
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out.squeeze()

# モデル初期化
input_size = len(feature_cols)
model = StockLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print(f"モデル構築完了！")
print(f"  入力サイズ: {input_size}")
print(f"  隠れ層サイズ: {HIDDEN_SIZE}")
print(f"  LSTM層数: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"\nモデル構造:")
print(model)

# パラメータ数
total_params = sum(p.numel() for p in model.parameters())
print(f"\n総パラメータ数: {total_params:,}")

# ===== 6. 訓練 =====
print("\n" + "=" * 80)
print("6. モデル訓練")
print("=" * 80)

# 不均衡データ対策：重み付き損失
pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])]).to(DEVICE)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
best_val_loss = float('inf')

print(f"エポック数: {EPOCHS}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"学習率: {LEARNING_RATE}")
print(f"\n訓練開始...")

for epoch in range(EPOCHS):
    # 訓練
    model.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 検証
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # ベストモデル保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print(f"\n訓練完了！ベストモデル保存済み（best_model.pth）")

# ===== 7. 評価 =====
print("\n" + "=" * 80)
print("7. モデル評価")
print("=" * 80)

# ベストモデルをロード
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def evaluate_model(model, data_loader, data_name="Test"):
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # 閾値0.5で二値化
    y_pred = (predictions > 0.5).astype(int)
    
    # 評価指標
    acc = accuracy_score(targets, y_pred)
    prec = precision_score(targets, y_pred, zero_division=0)
    rec = recall_score(targets, y_pred, zero_division=0)
    f1 = f1_score(targets, y_pred, zero_division=0)
    cm = confusion_matrix(targets, y_pred)
    
    print(f"\n【{data_name}データ評価結果】")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  混同行列:")
    print(f"                予測0   予測1")
    print(f"    実際0    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"    実際1    {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    return predictions, targets, y_pred

# 各データセットで評価
print("\n" + "-" * 80)
train_pred, train_target, train_pred_binary = evaluate_model(model, train_loader, "訓練")
val_pred, val_target, val_pred_binary = evaluate_model(model, val_loader, "検証")
test_pred, test_target, test_pred_binary = evaluate_model(model, test_loader, "テスト")

# ===== 8. 結果の可視化 =====
print("\n" + "=" * 80)
print("8. 結果の可視化")
print("=" * 80)

# グラフ1: 学習曲線
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, EPOCHS+1), train_losses, label='訓練Loss', marker='o')
ax.plot(range(1, EPOCHS+1), val_losses, label='検証Loss', marker='s')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('学習曲線（Loss）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# グラフ2: 混同行列（テストデータ）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('混同行列', fontsize=16, fontweight='bold')

for idx, (pred, target, name) in enumerate([
    (train_pred_binary, train_target, '訓練'),
    (val_pred_binary, val_target, '検証'),
    (test_pred_binary, test_target, 'テスト')
]):
    cm = confusion_matrix(target, pred)
    im = axes[idx].imshow(cm, cmap='Blues')
    axes[idx].set_title(f'{name}データ')
    axes[idx].set_xlabel('予測')
    axes[idx].set_ylabel('実際')
    axes[idx].set_xticks([0, 1])
    axes[idx].set_yticks([0, 1])
    axes[idx].set_xticklabels(['基準超=0', '基準超=1'])
    axes[idx].set_yticklabels(['基準超=0', '基準超=1'])
    
    # 数値を表示
    for i in range(2):
        for j in range(2):
            axes[idx].text(j, i, str(cm[i, j]), 
                          ha='center', va='center', 
                          color='white' if cm[i, j] > cm.max()/2 else 'black',
                          fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.show()

# グラフ3: 予測確率の分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('予測確率の分布', fontsize=16, fontweight='bold')

for idx, (pred, target, name) in enumerate([
    (train_pred, train_target, '訓練'),
    (val_pred, val_target, '検証'),
    (test_pred, test_target, 'テスト')
]):
    axes[idx].hist(pred[target==0], bins=30, alpha=0.6, label='実際0', color='blue')
    axes[idx].hist(pred[target==1], bins=30, alpha=0.6, label='実際1', color='red')
    axes[idx].axvline(x=0.5, color='green', linestyle='--', label='閾値0.5')
    axes[idx].set_title(f'{name}データ')
    axes[idx].set_xlabel('予測確率')
    axes[idx].set_ylabel('頻度')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
print(f"\n保存されたファイル:")
print(f"  - best_model.pth: 学習済みモデル")