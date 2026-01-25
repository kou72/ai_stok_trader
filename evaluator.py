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
        
        # テストデータの場合は基準超=1の正答率のみ表示
        if name == 'テスト':
            print(f"\n{'='*80}")
            print(f"【{name}データ評価結果】")
            print(f"{'='*80}")
            
            actual_1_total = np.sum(targets == 1)  # 実際に基準超=1だった数
            correct_1 = cm[1][1]  # 正しく予測できた数
            recall = correct_1 / max(actual_1_total, 1)
            
            print(f"\n  🎯 基準超=1の予測結果:")
            print(f"    実際に上昇した日数:     {actual_1_total} 日")
            print(f"    正しく予測できた日数:   {correct_1} 日")
            print(f"    正答率:                 {recall:.4f} ({recall*100:.2f}%)")
            
            print(f"\n  参考:")
            print(f"    全体の正解率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            
        else:
            # 訓練・検証データは詳細を表示
            print(f"\n{'='*80}")
            print(f"【{name}データ評価結果】")
            print(f"{'='*80}")
            print(f"  正解率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  適合率 (Precision): {metrics['precision']:.4f}")
            print(f"  再現率 (Recall):    {metrics['recall']:.4f}")
            print(f"  F1スコア:           {metrics['f1']:.4f}")
            print(f"\n  混同行列:")
            print(f"                予測0   予測1")
            print(f"    実際0    {cm[0][0]:6d}  {cm[0][1]:6d}")
            print(f"    実際1    {cm[1][0]:6d}  {cm[1][1]:6d}")
            
            # わかりやすい正答率の表示
            total = len(targets)
            correct_0 = cm[0][0]
            correct_1 = cm[1][1]
            
            print(f"\n  詳細:")
            print(f"    基準超=0の正解数: {correct_0}/{np.sum(targets==0)} ({correct_0/max(np.sum(targets==0),1)*100:.2f}%)")
            print(f"    基準超=1の正解数: {correct_1}/{np.sum(targets==1)} ({correct_1/max(np.sum(targets==1),1)*100:.2f}%)")
            print(f"    全体の正解数:     {correct_0+correct_1}/{total} ({(correct_0+correct_1)/total*100:.2f}%)")
    
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
        
        # グラフ1: 学習曲線
        self._plot_learning_curve(train_losses, val_losses, epochs)
        
        # グラフ2: 混同行列
        self._plot_confusion_matrices(results)
        
        # グラフ3: 予測確率の分布
        self._plot_prediction_distributions(results)
        
        # グラフ4: 正答率のサマリー
        self._plot_accuracy_summary(results)
    
    @staticmethod
    def _plot_learning_curve(train_losses, val_losses, epochs):
        """学習曲線"""
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
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
    def _plot_confusion_matrices(results):
        """混同行列"""
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('混同行列', fontsize=16, fontweight='bold')
        
        for idx, name in enumerate(['訓練', '検証', 'テスト']):
            cm = results[name]['metrics']['confusion_matrix']
            im = axes[idx].imshow(cm, cmap='Blues')
            axes[idx].set_title(f'{name}データ')
            axes[idx].set_xlabel('予測')
            axes[idx].set_ylabel('実際')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['基準超=0', '基準超=1'])
            axes[idx].set_yticklabels(['基準超=0', '基準超=1'])
            
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(cm[i, j]),
                                  ha='center', va='center',
                                  color='white' if cm[i, j] > cm.max()/2 else 'black',
                                  fontsize=14, fontweight='bold')
            
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_prediction_distributions(results):
        """予測確率の分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('予測確率の分布', fontsize=16, fontweight='bold')
        
        for idx, name in enumerate(['訓練', '検証', 'テスト']):
            pred = results[name]['predictions']
            target = results[name]['targets']
            
            axes[idx].hist(pred[target==0], bins=30, alpha=0.6, label='実際0', color='blue')
            axes[idx].hist(pred[target==1], bins=30, alpha=0.6, label='実際1', color='red')
            axes[idx].axvline(x=0.5, color='green', linestyle='--', label='閾値0.5')
            axes[idx].set_title(f'{name}データ')
            axes[idx].set_xlabel('予測確率')
            axes[idx].set_ylabel('頻度')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_accuracy_summary(results):
        """正答率のサマリー"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # グラフ1: 各指標の比較
        datasets = ['訓練', '検証', 'テスト']
        metrics_names = ['正解率', '適合率', '再現率', 'F1']
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, metric_key in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            values = [results[name]['metrics'][metric_key] for name in datasets]
            ax1.bar(x + i*width, values, width, label=metrics_names[i])
        
        ax1.set_xlabel('データセット', fontsize=12)
        ax1.set_ylabel('スコア', fontsize=12)
        ax1.set_title('評価指標の比較', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)
        
        # 数値ラベル
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.3f', fontsize=8)
        
        # グラフ2: 正答率の内訳
        for idx, name in enumerate(datasets):
            cm = results[name]['metrics']['confusion_matrix']
            total = cm.sum()
            correct_0 = cm[0][0]
            correct_1 = cm[1][1]
            incorrect = cm[0][1] + cm[1][0]
            
            ax2.bar(idx, correct_0, label='基準超=0 正解' if idx==0 else '', color='lightblue')
            ax2.bar(idx, correct_1, bottom=correct_0, label='基準超=1 正解' if idx==0 else '', color='lightcoral')
            ax2.bar(idx, incorrect, bottom=correct_0+correct_1, label='不正解' if idx==0 else '', color='lightgray')
            
            # 正解率を上部に表示
            accuracy = (correct_0 + correct_1) / total
            ax2.text(idx, total + 5, f'{accuracy*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
        
        ax2.set_xlabel('データセット', fontsize=12)
        ax2.set_ylabel('サンプル数', fontsize=12)
        ax2.set_title('正答率の内訳', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()