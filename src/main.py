"""
メイン実行ファイル
"""
import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import time

from config import Config
from data_processor import DataProcessor
from model import StockDataset, StockLSTM
from trainer import Trainer
from evaluator import Evaluator


def save_config_to_json(result_dir):
    """
    設定パラメータをJSONファイルに保存
    
    Parameters:
    -----------
    result_dir : str
        結果保存ディレクトリ
    """
    config_dict = {
        'TIME_STEP': Config.TIME_STEP,
        'TRAIN_RATIO': Config.TRAIN_RATIO,
        'VAL_RATIO': Config.VAL_RATIO,
        'TEST_RATIO': Config.TEST_RATIO,
        'FEATURE_COLS': Config.FEATURE_COLS,
        'HIDDEN_SIZE': Config.HIDDEN_SIZE,
        'NUM_LAYERS': Config.NUM_LAYERS,
        'DROPOUT': Config.DROPOUT,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'EPOCHS': Config.EPOCHS,
        'LEARNING_RATE': Config.LEARNING_RATE,
        'DEVICE': str(Config.DEVICE)
    }
    
    config_path = os.path.join(result_dir, 'config.json')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"設定ファイル保存: {config_path}")
    return config_path


def save_results_to_json(results, result_dir):
    """
    最終的な的中率をJSONファイルに保存
    
    Parameters:
    -----------
    results : dict
        各データセットの的中率
    result_dir : str
        結果保存ディレクトリ
    """
    results_path = os.path.join(result_dir, 'results.json')
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"結果ファイル保存: {results_path}")
    return results_path


def save_history_to_csv(train_losses, val_losses, train_precisions, val_precisions, result_dir):
    """
    損失と的中率の推移をCSVファイルに保存
    
    Parameters:
    -----------
    train_losses, val_losses : list
        訓練・検証のLoss履歴
    train_precisions, val_precisions : list
        訓練・検証の的中率履歴
    result_dir : str
        結果保存ディレクトリ
    """
    # 損失の推移をCSV保存
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_path = os.path.join(result_dir, 'loss_history.csv')
    loss_df.to_csv(loss_path, index=False, encoding='utf-8')
    print(f"損失履歴保存: {loss_path}")
    
    # 的中率の推移をCSV保存
    precision_df = pd.DataFrame({
        'epoch': range(1, len(train_precisions) + 1),
        'train_precision': train_precisions,
        'val_precision': val_precisions
    })
    precision_path = os.path.join(result_dir, 'precision_history.csv')
    precision_df.to_csv(precision_path, index=False, encoding='utf-8')
    print(f"的中率履歴保存: {precision_path}")
    
    return loss_path, precision_path


def save_time_log_to_json(time_log, result_dir):
    """
    処理時間をJSONファイルに保存
    
    Parameters:
    -----------
    time_log : dict
        各処理の時間記録
    result_dir : str
        結果保存ディレクトリ
    """
    time_path = os.path.join(result_dir, 'time_log.json')
    
    with open(time_path, 'w', encoding='utf-8') as f:
        json.dump(time_log, f, indent=4, ensure_ascii=False)
    
    print(f"処理時間保存: {time_path}")
    return time_path


def format_time(seconds):
    """
    秒数を時:分:秒形式に変換
    
    Parameters:
    -----------
    seconds : float
        秒数
    
    Returns:
    --------
    str : フォーマットされた時間文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}時間{minutes}分{secs:.2f}秒"
    elif minutes > 0:
        return f"{minutes}分{secs:.2f}秒"
    else:
        return f"{secs:.2f}秒"


def calculate_final_precision(results):
    """
    最終的な的中率を計算
    
    Parameters:
    -----------
    results : dict
        評価結果
    
    Returns:
    --------
    precision_dict : dict
        各データセットの的中率
    """
    precision_dict = {}
    
    for name in ['訓練', '検証', 'テスト']:
        predictions = results[name]['predictions']
        targets = results[name]['targets']
        
        pred_binary = (predictions > 0.5).astype(int)
        predicted_positive = pred_binary == 1
        n_predicted_positive = np.sum(predicted_positive)
        
        if n_predicted_positive == 0:
            precision = 0.0
        else:
            true_positive = np.sum((pred_binary == 1) & (targets == 1))
            precision = float(true_positive / n_predicted_positive)
        
        precision_dict[name] = {
            'predicted_count': int(n_predicted_positive),
            'correct_count': int(np.sum((pred_binary == 1) & (targets == 1))),
            'precision': precision,
            'precision_percent': f'{precision*100:.2f}%'
        }
    
    return precision_dict


def main(csv_path):
    """
    メイン処理
    
    Parameters:
    -----------
    csv_path : str
        CSVファイルのパス
    """
    # 全体の開始時間
    total_start_time = time.time()
    
    print("=" * 80)
    print("LSTM株価予測モデル")
    print("=" * 80)
    print(f"使用デバイス: {Config.DEVICE}\n")
    
    # タイムスタンプ生成
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 結果保存ディレクトリを作成
    result_dir = os.path.join('result', timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"結果保存先: {result_dir}\n")
    
    # モデル保存パスを生成
    os.makedirs('model', exist_ok=True)
    model_save_path = f'model/best_model_{timestamp}.pth'
    print(f"モデル保存先: {model_save_path}\n")
    
    # 時間記録用の辞書
    time_log = {}
    
    # 設定をJSON保存
    save_config_to_json(result_dir)
    
    # データ処理
    processor = DataProcessor(Config.FEATURE_COLS)
    
    # 1. CSVデータ読み込み
    start_time = time.time()
    df = processor.load_csv(csv_path)
    time_log['1_load_csv'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 2. データ前処理・正規化
    start_time = time.time()
    df_normalized = processor.preprocess_and_normalize(df)
    time_log['2_preprocess_and_normalize'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 3. シーケンス作成
    start_time = time.time()
    X, y, dates = processor.create_sequences(df_normalized, Config.TIME_STEP)
    time_log['3_create_sequences'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 4. 訓練/検証/テスト分割
    start_time = time.time()
    splits = processor.split_data(X, y, Config.TRAIN_RATIO, Config.VAL_RATIO)
    time_log['4_split_data'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # DataLoader作成
    start_time = time.time()
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
    time_log['5_create_dataloader'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 5. LSTMモデル構築
    print("\n" + "=" * 80)
    print("5. LSTMモデル構築")
    print("=" * 80)
    
    start_time = time.time()
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
    time_log['6_build_model'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 6. 訓練
    start_time = time.time()
    trainer = Trainer(model, Config.DEVICE, Config.LEARNING_RATE)
    train_losses, val_losses, train_precisions, val_precisions = trainer.train(
        train_loader, val_loader, Config.EPOCHS, model_save_path
    )
    time_log['7_train'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # ベストモデルをロード
    model.load_state_dict(torch.load(model_save_path))
    
    # 7. 評価
    start_time = time.time()
    evaluator = Evaluator(model, Config.DEVICE)
    results = evaluator.evaluate(train_loader, val_loader, test_loader)
    time_log['8_evaluate'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 最終的な的中率を計算
    precision_dict = calculate_final_precision(results)
    
    # 結果をJSON保存
    save_results_to_json(precision_dict, result_dir)
    
    # 損失と的中率の推移をCSV保存
    save_history_to_csv(train_losses, val_losses, train_precisions, val_precisions, result_dir)
    
    # 8. 結果の可視化
    start_time = time.time()
    evaluator.visualize(results, train_losses, val_losses, train_precisions, val_precisions, Config.EPOCHS)
    time_log['9_visualize'] = {
        'seconds': time.time() - start_time,
        'formatted': format_time(time.time() - start_time)
    }
    
    # 全体の処理時間
    total_time = time.time() - total_start_time
    time_log['total'] = {
        'seconds': total_time,
        'formatted': format_time(total_time)
    }
    
    # 処理時間をJSON保存
    save_time_log_to_json(time_log, result_dir)
    
    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print(f"\n処理時間:")
    print(f"  1. CSVデータ読み込み:      {time_log['1_load_csv']['formatted']}")
    print(f"  2. データ前処理・正規化:   {time_log['2_preprocess_and_normalize']['formatted']}")
    print(f"  3. シーケンス作成:         {time_log['3_create_sequences']['formatted']}")
    print(f"  4. データ分割:             {time_log['4_split_data']['formatted']}")
    print(f"  5. DataLoader作成:         {time_log['5_create_dataloader']['formatted']}")
    print(f"  6. モデル構築:             {time_log['6_build_model']['formatted']}")
    print(f"  7. 訓練:                   {time_log['7_train']['formatted']}")
    print(f"  8. 評価:                   {time_log['8_evaluate']['formatted']}")
    print(f"  9. 可視化:                 {time_log['9_visualize']['formatted']}")
    print(f"  ----------------------------------------")
    print(f"  合計:                      {time_log['total']['formatted']}")
    
    print(f"\n保存されたファイル:")
    print(f"  {result_dir}/")
    print(f"    ├── best_model.pth")
    print(f"    ├── config.json")
    print(f"    ├── results.json")
    print(f"    ├── loss_history.csv")
    print(f"    ├── precision_history.csv")
    print(f"    └── time_log.json")
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