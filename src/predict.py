"""
学習済みモデルで予測・評価
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import glob
import os
import matplotlib.pyplot as plt

from config import Config
from data_processor import DataProcessor
from model import StockDataset, StockLSTM
from evaluator import Evaluator


def get_latest_model(model_dir='model'):
    """
    最新のモデルファイルを取得
    
    Parameters:
    -----------
    model_dir : str
        モデルディレクトリ
    
    Returns:
    --------
    str : 最新のモデルファイルパス
    """
    model_files = glob.glob(os.path.join(model_dir, 'best_model_*.pth'))
    
    if not model_files:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_dir}")
    
    # タイムスタンプでソートして最新を取得
    latest_model = sorted(model_files)[-1]
    return latest_model


def main(csv_path, model_path):
    """
    学習済みモデルで予測・評価
    
    Parameters:
    -----------
    csv_path : str
        CSVファイルのパス
    model_path : str
        学習済みモデルのパス（'latest'の場合は最新モデルを使用）
    """
    print("=" * 80)
    print("学習済みモデルで予測・評価")
    print("=" * 80)
    print(f"使用デバイス: {Config.DEVICE}")
    
    # モデルパスの決定
    if model_path == 'latest':
        try:
            model_path = get_latest_model()
            print(f"最新モデルを使用: {model_path}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    else:
        print(f"指定モデル: {model_path}\n")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 5. モデル構築
    print("\n" + "=" * 80)
    print("5. モデルロード")
    print("=" * 80)
    
    model = StockLSTM(
        input_size=len(Config.FEATURE_COLS),
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # 学習済みモデルをロード
    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        print(f"✅ モデルをロードしました: {model_path}")
    except FileNotFoundError:
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    except Exception as e:
        print(f"❌ モデルのロードに失敗: {e}")
        return
    
    # 6. 評価
    evaluator = Evaluator(model, Config.DEVICE)
    results = evaluator.evaluate(train_loader, val_loader, test_loader)
    
    # 7. モデルが1と予測したときの的中率を計算・表示
    print("\n" + "=" * 80)
    print("7. モデルが「1」と予測したときの的中率")
    print("=" * 80)
    
    plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    datasets = ['訓練', '検証', 'テスト']
    precisions = []
    predicted_counts = []
    correct_counts = []
    
    for name in datasets:
        predictions = results[name]['predictions']
        targets = results[name]['targets']
        
        # 予測クラス（閾値0.5）
        pred_binary = (predictions > 0.5).astype(int)
        
        # モデルが1と予測したケース
        predicted_positive = pred_binary == 1
        n_predicted_positive = np.sum(predicted_positive)
        
        # その中で実際に1だったケース
        true_positive = np.sum((pred_binary == 1) & (targets == 1))
        
        # 的中率（適合率）
        precision = true_positive / n_predicted_positive if n_predicted_positive > 0 else 0
        
        precisions.append(precision)
        predicted_counts.append(n_predicted_positive)
        correct_counts.append(true_positive)
        
        print(f"\n【{name}データ】")
        print(f"  モデルが「1」と予測した数: {n_predicted_positive}件")
        print(f"  実際に「1」だった数: {true_positive}件")
        print(f"  的中率: {precision:.4f} ({precision*100:.2f}%)")
    
    # 8. 可視化
    print("\n" + "=" * 80)
    print("8. 結果の可視化")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(datasets, precisions, color=['skyblue', 'lightgreen', 'lightcoral'])
    
    ax.set_ylabel('的中率', fontsize=14)
    ax.set_title('モデルが「上昇する（1）」と予測したときの的中率', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 数値ラベル
    for i, (bar, precision) in enumerate(zip(bars, precisions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{precision*100:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 予測数を棒の下に表示
        ax.text(bar.get_x() + bar.get_width()/2., -0.05,
                f'{predicted_counts[i]}件中\n{correct_counts[i]}件的中',
                ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\n完了！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='学習済みモデルで予測・評価')
    parser.add_argument(
        '--csv',
        type=str,
        default='data/csv_20260126_003947_20',
        help='CSVファイルのパスまたはフォルダパス'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='latest',
        help='学習済みモデルのパス（"latest"で最新モデル使用）'
    )
    
    args = parser.parse_args()
    main(args.csv, args.model)