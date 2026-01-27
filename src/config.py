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

    # 基準超判定の閾値（上昇率が何%以上で基準超=1とするか）
    # 例: 1.0 → 終値が始値より1%以上上昇していれば基準超=1
    PRICE_INCREASE_THRESHOLD = 1.0

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
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # デバイス
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ファイルパス
    MODEL_SAVE_PATH = 'model/best_model.pth'