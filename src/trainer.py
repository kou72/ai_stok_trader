"""
モデルの訓練
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """モデル訓練クラス"""
    
    def __init__(self, model, device, learning_rate=0.001):
        """
        Parameters:
        -----------
        model : nn.Module
            訓練するモデル
        device : torch.device
            使用デバイス
        learning_rate : float
            学習率
        """
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
        self.val_losses = []
    
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
        train_losses, val_losses : list
            訓練・検証のLoss履歴
        """
        print("\n" + "=" * 80)
        print("6. モデル訓練")
        print("=" * 80)
        
        print(f"エポック数: {epochs}")
        print(f"訓練開始...\n")
        
        best_val_loss = float('inf')
        
        # エポックごとの進捗バー
        epoch_bar = tqdm(range(epochs), desc="訓練進捗", unit="epoch")
        
        for epoch in epoch_bar:
            # 訓練
            train_loss = self._train_epoch(train_loader, epoch+1, epochs)
            self.train_losses.append(train_loss)
            
            # 検証
            val_loss = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # ベストモデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                best_mark = " ⭐"
            else:
                best_mark = ""
            
            # 進捗バーの説明を更新
            epoch_bar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Best': f'{best_val_loss:.4f}'
            })
            
            # ログ出力
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}{best_mark}")
        
        print(f"\n訓練完了！ベストモデル保存済み（{model_save_path}）")
        
        return self.train_losses, self.val_losses
    
    def _train_epoch(self, train_loader, current_epoch, total_epochs):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        
        # バッチごとの進捗バー
        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {current_epoch}/{total_epochs} [Train]",
            leave=False,
            unit="batch"
        )
        
        for X_batch, y_batch in batch_bar:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 進捗バーの説明を更新
            batch_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0
        
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
                
                # 進捗バーの説明を更新
                batch_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
