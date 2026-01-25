"""
設定パラメータ
"""
import torch

class Config:
    """モデル訓練の設定"""
    
    # データ設定
    TIME_STEP = 480
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 特徴量
    FEATURE_COLS = [
        '始値', '高値', '安値', '終値', '出来高',
        '上昇率', 'MA5', 'MA20', 'MA60',
        'volatility_20', 'RSI', '日中変動率', '前日比'
    ]
    
    # モデル設定
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # 訓練設定
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # デバイス
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ファイルパス
    MODEL_SAVE_PATH = 'best_model.pth'