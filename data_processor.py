"""
データ読み込み・前処理・シーケンス作成
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """株価データの前処理とシーケンス作成"""
    
    def __init__(self, feature_cols):
        """
        Parameters:
        -----------
        feature_cols : list
            使用する特徴量のリスト
        """
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
    
    def load_csv(self, csv_path):
        """
        1. CSVデータ読み込み
        
        Parameters:
        -----------
        csv_path : str
            CSVファイルのパス
        
        Returns:
        --------
        df : DataFrame
            読み込んだデータ
        """
        print("=" * 80)
        print("1. CSVデータ読み込み")
        print("=" * 80)
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"ファイルパス: {csv_path}")
        print(f"元データ件数: {len(df)}")
        
        return df
    
    def preprocess_and_normalize(self, df):
        """
        2. データ前処理・正規化
        
        Parameters:
        -----------
        df : DataFrame
            元データ
        
        Returns:
        --------
        df_normalized : DataFrame
            前処理・正規化済みデータ
        """
        print("\n" + "=" * 80)
        print("2. データ前処理・正規化")
        print("=" * 80)
        
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
        df = self._add_features(df)
        df = df.dropna()
        print(f"特徴量追加後: {len(df)} 件")
        
        # 正規化
        df_normalized = df.copy()
        df_normalized[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        
        print(f"正規化完了！特徴量数: {len(self.feature_cols)}")
        
        return df_normalized
    
    def _add_features(self, df):
        """特徴量エンジニアリング"""
        # 移動平均
        df['MA5'] = df['終値'].rolling(window=5).mean()
        df['MA20'] = df['終値'].rolling(window=20).mean()
        df['MA60'] = df['終値'].rolling(window=60).mean()
        
        # ボラティリティ
        df['volatility_20'] = df['上昇率'].rolling(window=20).std()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['終値'])
        
        # その他
        df['日中変動率'] = (df['高値'] - df['安値']) / df['始値']
        df['前日比'] = df['終値'].pct_change()
        
        return df
    
    @staticmethod
    def _calculate_rsi(series, period=14):
        """RSIの計算"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_sequences(self, df, time_step):
        """
        3. シーケンス作成
        
        Parameters:
        -----------
        df : DataFrame
            正規化済みデータ
        time_step : int
            時系列の長さ
        
        Returns:
        --------
        X, y, dates : ndarray, ndarray, list
            シーケンスデータ
        """
        print("\n" + "=" * 80)
        print(f"3. シーケンス作成（time_step={time_step}）")
        print("=" * 80)
        
        X, y, dates = [], [], []
        
        for i in range(len(df) - time_step):
            X.append(df[self.feature_cols].iloc[i:i+time_step].values)
            y.append(df['基準超'].iloc[i+time_step])
            dates.append(df['日付'].iloc[i+time_step])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"シーケンス作成完了！")
        print(f"X.shape: {X.shape} (サンプル数, 時系列長, 特徴量数)")
        print(f"y.shape: {y.shape}")
        print(f"基準超=1の割合: {np.sum(y==1)/len(y)*100:.2f}%")
        
        return X, y, dates
    
    def split_data(self, X, y, train_ratio, val_ratio):
        """
        4. 訓練/検証/テスト分割
        
        Parameters:
        -----------
        X, y : ndarray
            シーケンスデータ
        train_ratio, val_ratio : float
            分割比率
        
        Returns:
        --------
        splits : dict
            分割されたデータ
        """
        print("\n" + "=" * 80)
        print("4. 訓練/検証/テスト分割")
        print("=" * 80)
        
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }
        
        print(f"訓練: {len(splits['X_train'])} サンプル")
        print(f"検証: {len(splits['X_val'])} サンプル")
        print(f"テスト: {len(splits['X_test'])} サンプル")
        
        return splits