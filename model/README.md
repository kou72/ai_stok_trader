# Model Directory

このディレクトリには、学習済みの株価予測モデルが保存されます。

## ファイル形式

学習済みモデルは `.pth` 形式で保存されます：
- `best_model_YYYYMMDD_HHMMSS.pth` - 学習時に最良の性能を示したモデル

## 使用方法

学習済みモデルをロードして予測に使用する場合：

```python
import torch
from model.stock_predictor import StockPredictor

# モデルの初期化
model = StockPredictor(input_size=your_input_size, hidden_size=your_hidden_size)

# 学習済みモデルのロード
model.load_state_dict(torch.load('model/best_model_YYYYMMDD_HHMMSS.pth'))
model.eval()
```

## 注意事項

- このディレクトリ内のモデルファイル（`.pth`）はGitの管理対象外です
- モデルファイルは容量が大きいため、リポジトリには含めません
- 学習を実行すると、新しいモデルファイルが自動的にこのディレクトリに保存されます
