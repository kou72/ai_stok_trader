"""
メイン実行ファイル
"""
import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from data_processor import DataProcessor
from model import StockDataset, StockLSTM
from trainer import Trainer
from evaluator import Evaluator


def main(csv_path):
    """
    メイン処理
    
    Parameters:
    -----------
    csv_path : str
        CSVファイルのパス
    """
    print("=" * 80)
    print("LSTM株価予測モデル")
    print("=" * 80)
    print(f"使用デバイス: {Config.DEVICE}\n")
    
    # データ処理
    processor = DataProcessor(Config.FEATURE_COLS)
    
    # 1. CSVデータ読み込み
    df = processor.load_csv(csv_path)
    
    # 2. データ前処理・正規化
    df_normalized = processor.preprocess_and_normalize(df)
    
    # 3. シーケンス作成
    X, y, dates = processor.create_sequences(df_normalized, Config.TIME_STEP)
    
    # 4. 訓練/検証/テスト分割
    splits = processor.split_data(X, y, Config.TRAIN_RATIO, Config.VAL_RATIO)
    
    # DataLoader作成
    train_dataset = StockDataset(splits['X_train'], splits['y_train'])
    val_dataset = StockDataset(splits['X_val'], splits['y_val'])
    test_dataset = StockDataset(splits['X_test'], splits['y_test'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 5. LSTMモデル構築
    print("\n" + "=" * 80)
    print("5. LSTMモデル構築")
    print("=" * 80)
    
    model = StockLSTM(
        input_size=len(Config.FEATURE_COLS),
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"モデル構築完了！")
    print(f"  入力サイズ: {len(Config.FEATURE_COLS)}")
    print(f"  隠れ層サイズ: {Config.HIDDEN_SIZE}")
    print(f"  LSTM層数: {Config.NUM_LAYERS}")
    print(f"  総パラメータ数: {total_params:,}")
    
    # 6. 訓練
    trainer = Trainer(model, Config.DEVICE, Config.LEARNING_RATE)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, Config.EPOCHS, Config.MODEL_SAVE_PATH
    )
    
    # ベストモデルをロード
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    # 7. 評価
    evaluator = Evaluator(model, Config.DEVICE)
    results = evaluator.evaluate(train_loader, val_loader, test_loader)
    
    # 8. 結果の可視化
    evaluator.visualize(results, train_losses, val_losses, Config.EPOCHS)
    
    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM株価予測モデル')
    parser.add_argument(
        '--csv',
        type=str,
        default='data/csv_20260126_003947',
        help='CSVファイルのパス'
    )
    
    args = parser.parse_args()
    main(args.csv)