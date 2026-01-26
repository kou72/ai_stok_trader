# Web Interface

株価予測モデル学習のWebインターフェースです。React + Tailwind CSSで構築されています。

## 概要

このWebアプリケーションは、`src/main.py`の学習処理をブラウザから実行できるモダンなインターフェースを提供します。

## 技術スタック

- **フロントエンド**: React 18 + Vite
- **スタイリング**: Tailwind CSS 3
- **バックエンド**: Flask (Python)

## 機能

- データセット（CSVディレクトリ）の選択
- ワンクリックで学習開始
- リアルタイムログ表示
- 学習の進捗状況表示
- 学習結果の保存先表示

## 必要なパッケージ

### Python
```bash
pip install flask
```

### Node.js
Node.js 18以上が必要です。

```bash
cd web/frontend
npm install
```

## セットアップ

### 1. フロントエンドのビルド

```bash
cd web/frontend
npm install
npm run build
```

これにより `web/static/dist/` ディレクトリにビルド成果物が生成されます。

### 2. Webサーバーの起動

プロジェクトのルートディレクトリから：

```bash
python web/app.py
```

または、webディレクトリ内で：

```bash
cd web
python app.py
```

### 3. ブラウザでアクセス

サーバーが起動したら、ブラウザで以下のURLにアクセスします：

```
http://localhost:5000
```

## 開発モード

フロントエンドの開発時は、Vite開発サーバーを使用できます：

```bash
cd web/frontend
npm run dev
```

開発サーバーは `http://localhost:5173` で起動し、ホットリロードが有効になります。
APIリクエストは自動的に `http://localhost:5000` にプロキシされます。

## ファイル構成

```
web/
├── frontend/              # Reactアプリケーション
│   ├── src/
│   │   ├── App.jsx       # メインコンポーネント
│   │   ├── main.jsx      # エントリーポイント
│   │   └── index.css     # Tailwind CSS
│   ├── index.html        # HTMLテンプレート
│   ├── package.json      # npm設定
│   ├── vite.config.js    # Vite設定
│   ├── tailwind.config.js # Tailwind設定
│   └── postcss.config.js  # PostCSS設定
├── static/
│   └── dist/             # ビルド成果物（自動生成）
├── app.py                # Flaskアプリケーション
└── README.md             # このファイル
```

## API エンドポイント

### GET /
Reactアプリを提供

### GET /csv_dirs
CSVディレクトリ一覧を取得

**レスポンス:**
```json
{
  "csv_dirs": ["csv_20260126_003947", "csv_20260126_001036"]
}
```

### POST /start_training
学習を開始

**リクエストボディ:**
```json
{
  "csv_dir": "csv_20260126_003947"
}
```

**レスポンス:**
```json
{
  "message": "学習を開始しました",
  "status": "started"
}
```

### GET /status
学習の現在の状態を取得

**レスポンス:**
```json
{
  "is_running": true,
  "start_time": "2026-01-27 10:30:00",
  "end_time": null,
  "result_dir": null,
  "new_logs": ["ログメッセージ1", "ログメッセージ2"]
}
```

### GET /logs
全ログを取得

**レスポンス:**
```json
{
  "logs": ["ログメッセージ1", "ログメッセージ2", "..."]
}
```

## 注意事項

- 学習の実行中は別の学習を開始できません
- ブラウザを閉じても学習は継続されます
- 複数のブラウザタブで同時にアクセス可能です（学習状態は共有されます）
- サーバーを停止すると、実行中の学習も停止します
- フロントエンドを変更した場合は `npm run build` でビルドし直してください

## トラブルシューティング

### ポート5000が既に使用されている

別のアプリケーションがポート5000を使用している場合、`app.py`の最後の行を変更してください：

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # 5000を8080などに変更
```

開発モードの場合は `vite.config.js` のプロキシ設定も変更が必要です。

### 学習が開始されない

- `data/csv_*` ディレクトリが存在するか確認してください
- `src/main.py` が正しく動作するか、コマンドラインから直接実行して確認してください

```bash
python src/main.py --csv data/csv_20260126_003947
```

### フロントエンドが表示されない

1. ビルドが完了しているか確認:
```bash
cd web/frontend
npm run build
```

2. `web/static/dist/` ディレクトリにファイルが生成されているか確認

### npm installでエラーが出る

Node.jsのバージョンを確認してください（18以上が必要）:
```bash
node --version
```

## 開発者向け

### スタイルのカスタマイズ

Tailwind CSSを使用しているため、[frontend/src/App.jsx](frontend/src/App.jsx) 内で直接クラス名を変更できます。

追加のカスタムスタイルが必要な場合は [frontend/tailwind.config.js](frontend/tailwind.config.js) を編集してください。

### 新しいコンポーネントの追加

`frontend/src/` ディレクトリに新しいコンポーネントを作成し、`App.jsx` からインポートしてください。

### APIエンドポイントの追加

[app.py](app.py) に新しいルートを追加してください。
