"""
LSTMモデルとDataset定義
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """株価データセット"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockLSTM(nn.Module):
    """株価予測用LSTMモデル"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        """
        5. LSTMモデル構築
        
        Parameters:
        -----------
        input_size : int
            入力特徴量数
        hidden_size : int
            隠れ層のサイズ
        num_layers : int
            LSTM層の数
        dropout : float
            ドロップアウト率
        """
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