"""
モデルの評価と可視化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch


class Evaluator:
    """モデル評価・可視化クラス"""
    
    def __init__(self, model, device):
        """
        Parameters:
        -----------
        model : nn.Module
            評価するモデル
        device : torch.device
            使用デバイス
        """
        self.model = model
        self.device = device
    
    def evaluate(self, train_loader, val_loader, test_loader):
        """
        7. 評価
        
        Parameters:
        -----------
        train_loader, val_loader, test_loader : DataLoader
            各データローダー
        
        Returns:
        --------
        results : dict
            評価結果
        """
        print("\n" + "=" * 80)
        print("7. モデル評価")
        print("=" * 80)
        
        self.model.eval()
        
        results = {}
        
        for name, loader in [('訓練', train_loader), ('検証', val_loader), ('テスト', test_loader)]:
            pred, target, pred_binary = self._evaluate_loader(loader)
            metrics = self._calculate_metrics(target, pred_binary)
            
            results[name] = {
                'predictions': pred,
                'targets': target,
                'predictions_binary': pred_binary,
                'metrics': metrics
            }
            
            self._print_metrics(name, metrics, target, pred_binary)
        
        return results
    
    def _evaluate_loader(self, data_loader):
        """データローダーで評価"""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        predictions_binary = (predictions > 0.5).astype(int)
        
        return predictions, targets, predictions_binary
    
    @staticmethod
    def _calculate_metrics(targets, predictions_binary):
        """評価指標の計算"""
        return {
            'accuracy': accuracy_score(targets, predictions_binary),
            'precision': precision_score(targets, predictions_binary, zero_division=0),
            'recall': recall_score(targets, predictions_binary, zero_division=0),
            'f1': f1_score(targets, predictions_binary, zero_division=0),
            'confusion_matrix': confusion_matrix(targets, predictions_binary)
        }
    
    @staticmethod
    def _print_metrics(name, metrics, targets, predictions_binary):
        """評価結果の表示"""
        cm = metrics['confusion_matrix']
        
        if name == 'テスト':
            # テストデータ: 基準超=1の正答率を詳細表示
            print(f"\n{'='*80}")
            print(f"【{name}データ評価結果】")
            print(f"{'='*80}")
            
            actual_1_total = np.sum(targets == 1)
            correct_1 = cm[1][1]
            recall = correct_1 / max(actual_1_total, 1)
            
            print(f"\n  🎯 基準超=1の予測結果:")
            print(f"    実際に上昇した日数:     {actual_1_total} 日")
            print(f"    正しく予測できた日数:   {correct_1} 日")
            print(f"    正答率 (再現率):        {recall:.4f} ({recall*100:.2f}%)")
            
            print(f"\n  混同行列:")
            print(f"                予測0   予測1")
            print(f"    実際0    {cm[0][0]:6d}  {cm[0][1]:6d}")
            print(f"    実際1    {cm[1][0]:6d}  {cm[1][1]:6d}")
            
            print(f"\n  参考:")
            print(f"    適合率 (Precision): {metrics['precision']:.4f} (予測が当たる確率)")
            print(f"    全体の正解率:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            
        else:
            # 訓練・検証データ: 簡易表示（過学習チェック用）
            recall = metrics['recall']
            print(f"\n【{name}データ】全体正解率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) | 基準超=1の正答率: {recall:.4f} ({recall*100:.2f}%)")
    
    def visualize(self, results, train_losses, val_losses, epochs):
        """
        8. 結果の可視化
        
        Parameters:
        -----------
        results : dict
            評価結果
        train_losses, val_losses : list
            訓練・検証のLoss履歴
        epochs : int
            エポック数
        """
        print("\n" + "=" * 80)
        print("8. 結果の可視化")
        print("=" * 80)
        
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # グラフ1: 学習曲線
        self._plot_learning_curve(train_losses, val_losses, epochs)
        
        # グラフ2: 基準超=1の正答率
        self._plot_recall_summary(results)
    
    @staticmethod
    def _plot_learning_curve(train_losses, val_losses, epochs):
        """学習曲線"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, epochs+1), train_losses, label='訓練Loss', marker='o')
        ax.plot(range(1, epochs+1), val_losses, label='検証Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('学習曲線（Loss）', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_recall_summary(results):
        """基準超=1の正答率（再現率）のサマリー"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # グラフ1: 基準超=1の正答率（再現率）の比較
        datasets = ['訓練', '検証', 'テスト']
        
        # 各データセットの基準超=1の正答率を計算
        recalls = []
        actual_counts = []
        correct_counts = []
        
        for name in datasets:
            cm = results[name]['metrics']['confusion_matrix']
            actual_1 = cm[1][0] + cm[1][1]  # 実際に1だった総数
            correct_1 = cm[1][1]  # 正しく予測
            recall = correct_1 / max(actual_1, 1)
            
            recalls.append(recall)
            actual_counts.append(actual_1)
            correct_counts.append(correct_1)
        
        x = np.arange(len(datasets))
        bars = ax1.bar(x, recalls, color=['skyblue', 'lightgreen', 'lightcoral'])
        
        ax1.set_xlabel('データセット', fontsize=12)
        ax1.set_ylabel('正答率（再現率）', fontsize=12)
        ax1.set_title('基準超=1の正答率', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 数値ラベル
        for i, (bar, recall) in enumerate(zip(bars, recalls)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{recall*100:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # グラフ2: 基準超=1の詳細（件数）
        x_pos = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, actual_counts, width, label='実際に上昇', color='gold')
        bars2 = ax2.bar(x_pos + width/2, correct_counts, width, label='正しく予測', color='lightcoral')
        
        ax2.set_xlabel('データセット', fontsize=12)
        ax2.set_ylabel('日数', fontsize=12)
        ax2.set_title('基準超=1の詳細', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 数値ラベル
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()