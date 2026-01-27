"""
データ読み込み・前処理・シーケンス作成
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class DataProcessor:
    """株価データの前処理とシーケンス作成"""

    def __init__(self, feature_cols, price_increase_threshold=1.0):
        """
        Parameters:
        -----------
        feature_cols : list
            使用する特徴量のリスト
        price_increase_threshold : float
            基準超の閾値（%）。上昇率がこの値以上なら基準超=1
        """
        self.feature_cols = feature_cols
        self.price_increase_threshold = price_increase_threshold
        self.scaler = StandardScaler()

    def load_csv(self, csv_path):
        """
        1. CSVデータ読み込み
        
        Parameters:
        -----------
        csv_path : str
            CSVファイルのパスまたはフォルダパス
        
        Returns:
        --------
        df : DataFrame
            読み込んだデータ
        """
        print("=" * 80)
        print("1. CSVデータ読み込み")
        print("=" * 80)
        
        import os
        import glob
        
        # フォルダかファイルか判定
        if os.path.isdir(csv_path):
            # フォルダの場合：全CSVファイルを読み込み
            csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
            
            print(f"フォルダパス: {csv_path}")
            print(f"CSVファイル数: {len(csv_files)}\n")
            
            df_list = []
            
            # 進捗バー付きでCSVファイルを読み込み
            for csv_file in tqdm(csv_files, desc="CSVファイル読み込み中", unit="ファイル"):
                try:
                    df_temp = pd.read_csv(csv_file, encoding='utf-8')
                    
                    # 銘柄コードを追加（ファイル名から取得）
                    stock_code = os.path.splitext(os.path.basename(csv_file))[0]
                    df_temp['銘柄コード'] = stock_code
                    
                    df_list.append(df_temp)
                except Exception as e:
                    tqdm.write(f"  警告: {csv_file} の読み込みに失敗: {e}")
            
            # 全データを結合
            df = pd.concat(df_list, ignore_index=True)
            print(f"\n読み込んだ銘柄数: {len(df_list)}")
            print(f"元データ件数: {len(df):,}")
            
        else:
            # ファイルの場合：1つのCSVを読み込み
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"ファイルパス: {csv_path}")
            print(f"元データ件数: {len(df):,}")
            
            # 銘柄コードを追加（ファイル名から取得）
            stock_code = os.path.splitext(os.path.basename(csv_path))[0]
            df['銘柄コード'] = stock_code
        
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
        numeric_cols = ['始値', '高値', '安値', '終値', '出来高']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 欠損値・異常値削除
        df = df.dropna()
        df = df[df['出来高'] > 0]
        df = df[(df['始値'] > 0) & (df['終値'] > 0)]
        df = df.sort_values(['銘柄コード', '日付']).reset_index(drop=True)
        
        print(f"前処理後: {len(df):,} 件\n")
        
        # 銘柄ごとに特徴量エンジニアリング
        stock_codes = df['銘柄コード'].unique()
        df_list = []
        
        for stock_code in tqdm(stock_codes, desc="特徴量エンジニアリング", unit="銘柄"):
            df_stock = df[df['銘柄コード'] == stock_code].copy()
            df_stock = self._add_features(df_stock)
            df_list.append(df_stock)
        
        df = pd.concat(df_list, ignore_index=True)
        df = df.dropna()
        print(f"\n特徴量追加後: {len(df):,} 件\n")
        
        # 銘柄ごとに正規化
        df_normalized_list = []
        
        for stock_code in tqdm(stock_codes, desc="正規化中", unit="銘柄"):
            df_stock = df[df['銘柄コード'] == stock_code].copy()
            
            if len(df_stock) == 0:
                continue
            
            # 銘柄ごとにScalerを作成
            scaler = StandardScaler()
            df_stock[self.feature_cols] = scaler.fit_transform(df_stock[self.feature_cols])
            
            df_normalized_list.append(df_stock)
        
        df_normalized = pd.concat(df_normalized_list, ignore_index=True)
        df_normalized = df_normalized.sort_values(['銘柄コード', '日付']).reset_index(drop=True)
        
        print(f"\n正規化完了！")
        print(f"  特徴量数: {len(self.feature_cols)}")
        print(f"  銘柄数: {df_normalized['銘柄コード'].nunique()}")
        print(f"  総データ件数: {len(df_normalized):,}")
        
        return df_normalized
    
    def _add_features(self, df):
        """特徴量エンジニアリング"""
        # 上昇率を計算（終値/始値 - 1）* 100 でパーセント表記
        df['上昇率'] = (df['終値'] / df['始値'] - 1) * 100

        # 基準超を計算（上昇率が閾値以上なら1）
        df['基準超'] = (df['上昇率'] >= self.price_increase_threshold).astype(int)

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
        3. シーケンス作成（NumPy最適化版）
        
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
        
        # 銘柄ごとにデータを分割
        stock_codes = df['銘柄コード'].unique()
        X_list, y_list, dates_list = [], [], []
        skipped_count = 0
        
        for stock_code in tqdm(stock_codes, desc="シーケンス作成中", unit="銘柄"):
            df_stock = df[df['銘柄コード'] == stock_code].reset_index(drop=True)
            
            if len(df_stock) <= time_step:
                skipped_count += 1
                continue
            
            # NumPy配列に変換
            features = df_stock[self.feature_cols].values
            targets = df_stock['基準超'].values
            dates = df_stock['日付'].values
            
            n_samples = len(features) - time_step
            n_features = features.shape[1]
            
            # 事前に配列を確保（高速化のポイント）
            X_stock = np.zeros((n_samples, time_step, n_features))
            y_stock = np.zeros(n_samples)
            
            # データを埋める
            for i in range(n_samples):
                X_stock[i] = features[i:i+time_step]
                y_stock[i] = targets[i+time_step]
            
            X_list.append(X_stock)
            y_list.append(y_stock)
            dates_list.extend(dates[time_step:])
        
        # 配列の結合（進捗表示付き）
        print("\n配列を結合中...")
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        print(f"\nシーケンス作成完了！")
        print(f"  対象銘柄数: {len(stock_codes) - skipped_count}/{len(stock_codes)}")
        print(f"  X.shape: {X.shape} (サンプル数, 時系列長, 特徴量数)")
        print(f"  y.shape: {y.shape}")
        print(f"  基準超=1の割合: {np.sum(y==1)/len(y)*100:.2f}%")
        
        return X, y, dates_list
    
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
        
        print(f"訓練: {len(splits['X_train']):,} サンプル")
        print(f"検証: {len(splits['X_val']):,} サンプル")
        print(f"テスト: {len(splits['X_test']):,} サンプル")
        
        return splits