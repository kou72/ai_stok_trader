"""
モデルの訓練
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Trainer:
    """モデル訓練クラス"""

    def __init__(self, model, device, learning_rate=0.001, progress_manager=None):
        """
        Parameters:
        -----------
        model : nn.Module
            訓練するモデル
        device : torch.device
            使用デバイス
        learning_rate : float
            学習率
        progress_manager : ProgressManager
            進捗管理オブジェクト
        """
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
        self.train_precisions = []  # 訓練時の的中率
        self.val_precisions = []    # 検証時の的中率
        self.progress_manager = progress_manager
    
    def train(self, train_loader, val_loader, epochs, model_save_path):
        """
        6. 訓練
        
        Parameters:
        -----------
        train_loader, val_loader : DataLoader
            訓練・検証データローダー
        epochs : int
            エポック数
        model_save_path : str
            モデル保存パス
        
        Returns:
        --------
        train_losses, val_losses, train_precisions, val_precisions : list
            訓練・検証のLoss履歴と的中率履歴
        """
        print(f"エポック数: {epochs}")
        print(f"訓練開始...\n")
        
        best_val_loss = float('inf')
        
        # エポックごとの進捗バー
        epoch_bar = tqdm(range(epochs), desc="訓練進捗", unit="epoch")
        
        total_batches = len(train_loader)

        for epoch in epoch_bar:
            # 訓練
            train_loss, train_precision = self._train_epoch(
                train_loader, epoch+1, epochs, total_batches
            )
            self.train_losses.append(train_loss)
            self.train_precisions.append(train_precision)

            # 検証
            val_loss, val_precision = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_precisions.append(val_precision)

            # ベストモデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)

            # 進捗バーの説明を更新
            epoch_bar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train Prec': f'{train_precision*100:.1f}%',
                'Val Prec': f'{val_precision*100:.1f}%'
            })

            # 進捗ファイルを更新（エポック終了時）
            if self.progress_manager:
                self.progress_manager.update_training(
                    epoch=epoch+1,
                    total_epochs=epochs,
                    batch=total_batches,
                    total_batches=total_batches,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_precision=train_precision,
                    val_precision=val_precision
                )

            # ログ出力
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Prec: {train_precision*100:.1f}%, Val Prec: {val_precision*100:.1f}%")
        
        print(f"\n訓練完了！ベストモデル保存済み（{model_save_path}）")
        
        return self.train_losses, self.val_losses, self.train_precisions, self.val_precisions
    
    def _train_epoch(self, train_loader, current_epoch, total_epochs, total_batches):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # バッチごとの進捗バー
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {current_epoch}/{total_epochs} [Train]",
            leave=False,
            unit="batch"
        )

        for batch_idx, (X_batch, y_batch) in enumerate(batch_bar):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 的中率計算用にデータを保存
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

            # 進捗バーの説明を更新
            batch_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # 進捗ファイルを更新（10バッチごと）
            if self.progress_manager and (batch_idx + 1) % 10 == 0:
                self.progress_manager.update_training(
                    epoch=current_epoch,
                    total_epochs=total_epochs,
                    batch=batch_idx + 1,
                    total_batches=total_batches
                )
        
        # 的中率を計算
        precision = self._calculate_precision(np.array(all_predictions), np.array(all_targets))
        
        return total_loss / len(train_loader), precision
    
    def _validate_epoch(self, val_loader):
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # バッチごとの進捗バー
        batch_bar = tqdm(
            val_loader,
            desc=f"  検証中",
            leave=False,
            unit="batch"
        )
        
        with torch.no_grad():
            for X_batch, y_batch in batch_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # 的中率計算用にデータを保存
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
                # 進捗バーの説明を更新
                batch_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 的中率を計算
        precision = self._calculate_precision(np.array(all_predictions), np.array(all_targets))
        
        return total_loss / len(val_loader), precision
    
    @staticmethod
    def _calculate_precision(predictions, targets):
        """
        モデルが1と予測したときの的中率（適合率）を計算
        
        Parameters:
        -----------
        predictions : ndarray
            予測確率
        targets : ndarray
            正解ラベル
        
        Returns:
        --------
        precision : float
            的中率
        """
        pred_binary = (predictions > 0.5).astype(int)
        predicted_positive = pred_binary == 1
        n_predicted_positive = np.sum(predicted_positive)
        
        if n_predicted_positive == 0:
            return 0.0
        
        true_positive = np.sum((pred_binary == 1) & (targets == 1))
        precision = true_positive / n_predicted_positive
        
        return precision