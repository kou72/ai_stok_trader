"""
進捗管理モジュール
JSONファイルを通じて学習の進捗状況を共有する
"""
import json
import os
from datetime import datetime


class ProgressManager:
    """学習進捗を管理するクラス"""

    SECTIONS = [
        'データ処理',
        '学習',
        '評価'
    ]

    def __init__(self, progress_file='progress.json'):
        """
        Parameters:
        -----------
        progress_file : str
            進捗ファイルのパス
        """
        self.progress_file = progress_file
        self.total_sections = len(self.SECTIONS)
        self.reset()

    def reset(self):
        """進捗をリセット"""
        self.progress = {
            'is_running': False,
            'current_section': 0,
            'total_sections': self.total_sections,
            'section_name': '',
            'section_percent': 0.0,
            'section_detail': '',
            'epoch': 0,
            'total_epochs': 0,
            'batch': 0,
            'total_batches': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_precision': 0.0,
            'val_precision': 0.0,
            'start_time': None,
            'updated_at': None,
            'running_params': None
        }
        self._save()

    def start(self, total_epochs=0, running_params=None):
        """学習開始"""
        self.progress['is_running'] = True
        self.progress['current_section'] = 0
        self.progress['total_epochs'] = total_epochs
        self.progress['start_time'] = datetime.now().isoformat()
        self.progress['updated_at'] = datetime.now().isoformat()
        self.progress['running_params'] = running_params
        self._save()

    def set_section(self, section_index, detail=''):
        """
        現在のセクションを設定

        Parameters:
        -----------
        section_index : int
            セクションのインデックス（0: データ処理, 1: 学習, 2: 評価）
        detail : str
            セクション内の詳細ステップ名
        """
        self.progress['current_section'] = section_index + 1
        self.progress['section_name'] = self.SECTIONS[section_index] if section_index < len(self.SECTIONS) else ''
        self.progress['section_detail'] = detail
        self.progress['section_percent'] = 0.0
        self.progress['updated_at'] = datetime.now().isoformat()
        self._save()

    def update_section_progress(self, percent, detail=''):
        """
        セクション内の進捗を更新

        Parameters:
        -----------
        percent : float
            セクション内の進捗率（0-100）
        detail : str
            詳細ステップ名
        """
        self.progress['section_percent'] = min(percent, 100.0)
        if detail:
            self.progress['section_detail'] = detail
        self.progress['updated_at'] = datetime.now().isoformat()
        self._save()

    def update_training(self, epoch, total_epochs, batch, total_batches,
                        train_loss=None, val_loss=None,
                        train_precision=None, val_precision=None):
        """
        訓練進捗を更新（学習セクション用）

        Parameters:
        -----------
        epoch : int
            現在のエポック
        total_epochs : int
            総エポック数
        batch : int
            現在のバッチ
        total_batches : int
            総バッチ数
        """
        self.progress['epoch'] = epoch
        self.progress['total_epochs'] = total_epochs
        self.progress['batch'] = batch
        self.progress['total_batches'] = total_batches

        if train_loss is not None:
            self.progress['train_loss'] = train_loss
        if val_loss is not None:
            self.progress['val_loss'] = val_loss
        if train_precision is not None:
            self.progress['train_precision'] = train_precision
        if val_precision is not None:
            self.progress['val_precision'] = val_precision

        # 学習セクションの進捗率を計算
        if total_epochs > 0 and total_batches > 0:
            epoch_progress = (epoch - 1 + batch / total_batches) / total_epochs
            self.progress['section_percent'] = epoch_progress * 100.0

        self.progress['updated_at'] = datetime.now().isoformat()
        self._save()

    def finish(self):
        """学習完了"""
        self.progress['is_running'] = False
        self.progress['current_section'] = self.total_sections
        self.progress['section_name'] = '完了'
        self.progress['section_percent'] = 100.0
        self.progress['updated_at'] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """進捗をファイルに保存"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"進捗ファイル保存エラー: {e}")

    @classmethod
    def load(cls, progress_file='progress.json'):
        """
        進捗ファイルを読み込む

        Parameters:
        -----------
        progress_file : str
            進捗ファイルのパス

        Returns:
        --------
        dict : 進捗情報
        """
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass

        return {
            'is_running': False,
            'current_section': 0,
            'total_sections': len(cls.SECTIONS),
            'section_name': '',
            'section_percent': 0.0,
            'section_detail': '',
            'epoch': 0,
            'total_epochs': 0,
            'batch': 0,
            'total_batches': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_precision': 0.0,
            'val_precision': 0.0,
            'start_time': None,
            'updated_at': None,
            'running_params': None
        }
