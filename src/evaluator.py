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
    
    def visualize(self, results, train_losses, val_losses, train_precisions, val_precisions, epochs):
        """
        8. 結果の可視化
        
        Parameters:
        -----------
        results : dict
            評価結果
        train_losses, val_losses : list
            訓練・検証のLoss履歴
        train_precisions, val_precisions : list
            訓練・検証の的中率履歴
        epochs : int
            エポック数
        """
        print("\n" + "=" * 80)
        print("8. 結果の可視化")
        print("=" * 80)
        
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 2軸グラフで損失と的中率を同時に表示
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左軸: Loss
        color1 = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', color=color1, fontsize=12)
        line1 = ax1.plot(range(1, epochs+1), train_losses, color=color1, marker='o', label='訓練Loss', linewidth=2)
        line2 = ax1.plot(range(1, epochs+1), val_losses, color='tab:cyan', marker='s', label='検証Loss', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 右軸: 的中率
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('的中率 (%)', color=color2, fontsize=12)
        line3 = ax2.plot(range(1, epochs+1), [p*100 for p in train_precisions], color=color2, marker='^', label='訓練的中率', linewidth=2)
        line4 = ax2.plot(range(1, epochs+1), [p*100 for p in val_precisions], color='tab:orange', marker='v', label='検証的中率', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 100)
        
        # 凡例を統合
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)
        
        plt.title('損失と的中率の推移', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()