# NILE - NeuroSymbolic Interactive Language Engine

NILEは日本語自然言語理解とシンボリック推論を組み合わせた実験的プロトタイプです。BERTベースの言語理解と知識グラフによる推論を統合し、日本語での自然な対話を実現します。

## ✨ 主な機能

### 🧠 高度な日本語処理
- **BERTベースの言語理解**: 日本語BERTモデルによる深い言語理解
- **エンティティ抽出**: 人物、動物、場所、概念などの自動抽出
- **関係性認識**: 複雑な日本語表現からの関係性抽出
- **時間表現処理**: 相対時間・絶対時間の正規化
- **曖昧性解消**: 文脈に基づく意味の曖昧性解消

### 🔗 シンボリック推論
- **知識グラフ**: NetworkXベースの効率的な知識表現
- **多様な推論タイプ**: 演繹・帰納・仮説・時間・因果推論
- **推論の可視化**: 推論過程のステップバイステップ表示
- **信頼度付き推論**: 確率的推論と信頼度評価

### 🚀 パフォーマンス最適化
- **並列処理**: マルチスレッド・マルチプロセス対応
- **インテリジェントキャッシュ**: LRU・時間ベースキャッシュ
- **効率的なデータ構造**: インデックス化された知識ベース
- **メモリ最適化**: 大規模データセット対応

### 🎨 多様なインターフェース
- **CLI**: コマンドラインインターフェース
- **Web UI**: Streamlitベースの美しいWebインターフェース
- **REST API**: FastAPIベースのAPI（開発中）
- **リアルタイム通信**: WebSocket対応（開発中）

## 📦 インストール

### 基本的なインストール

```bash
# リポジトリをクローン
git clone https://github.com/your-org/nile.git
cd nile

# 依存関係をインストール
pip install -e .
```

### 開発環境のセットアップ

```bash
# 開発用依存関係を含めてインストール
pip install -e ".[dev]"

# Web UI用依存関係
pip install -e ".[web]"

# GPU対応（CUDA環境）
pip install -e ".[gpu]"
```


```

## 🚀 使用方法

### コマンドラインインターフェース

```bash
# 基本的な使用
python -m nile.main

# または
python nile/main.py
```

### Webインターフェース

```bash
# Streamlitアプリを起動
streamlit run nile/ui/web_interface.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

### プログラムからの使用

```python
from nile import SymbolicEngine, KnowledgeBase

# エンジンの初期化
engine = SymbolicEngine("knowledge.json")

# 事実の登録
response = engine.process_input("猫は動物です")
print(response)  # "了解しました。猫と動物の関係を記録しました。"

# 質問
response = engine.process_input("猫は動物ですか？")
print(response)  # "はい、猫は動物です。"

# 推論の説明
explanation = engine.get_reasoning_explanation("猫は動物ですか？")
print(explanation)
```

## 📚 使用例

### 基本的な対話

```
質問を入力してください: 猫は動物です
了解しました。猫と動物の関係を記録しました。

質問を入力してください: 犬は動物です
了解しました。犬と動物の関係を記録しました。

質問を入力してください: 猫は動物ですか？
はい、猫は動物です。

質問を入力してください: 犬と猫の関係は？
はい、間接的な関係があります。
- 猫は動物と関連しています。
- 犬は動物と関連しています。
```

### 複雑な推論

```
質問を入力してください: 哺乳類は動物です
了解しました。哺乳類と動物の関係を記録しました。

質問を入力してください: 猫は哺乳類です
了解しました。猫と哺乳類の関係を記録しました。

質問を入力してください: 猫は動物ですか？
はい、猫は動物です。
推論過程：猫から動物への関係
1. 猫は哺乳類です。
2. 哺乳類は動物です。
結論：猫と動物は関連しています。
```

### 時間表現の処理

```
質問を入力してください: 今日は良い天気です
了解しました。今日と良い天気の関係を記録しました。

質問を入力してください: 今日の天気は？
はい、今日と良い天気の関係が存在します。
```

## 🏗️ アーキテクチャ

```
nile/
├── core/                    # コア機能
│   ├── nlp/                # 自然言語処理
│   │   ├── japanese_processor.py
│   │   ├── pattern_matcher.py
│   │   └── entity_extractor.py
│   ├── reasoning/          # 推論エンジン
│   │   ├── symbolic_engine.py
│   │   ├── knowledge_graph.py
│   │   └── inference_rules.py
│   └── storage/            # データストレージ
│       ├── knowledge_base.py
│       ├── fact_store.py
│       └── cache_manager.py
├── api/                    # API
│   ├── rest_api.py
│   └── websocket_api.py
├── ui/                     # ユーザーインターフェース
│   ├── web_interface.py
│   └── cli.py
├── tests/                  # テスト
└── config/                 # 設定
    └── settings.py
```

## 🧪 テスト

```bash
# 全テストを実行
pytest

# カバレッジ付きでテスト実行
pytest --cov=nile

# 特定のテストを実行
pytest nile/tests/test_knowledge_base.py

# 統合テストのみ実行
pytest -m integration
```

## 📊 パフォーマンス

- **処理速度**: 平均応答時間 < 1秒
- **メモリ使用量**: 基本構成で < 2GB
- **スケーラビリティ**: 10,000+ 事実の処理に対応
- **並列処理**: CPUコア数に応じた自動スケーリング

## 🔧 設定

設定ファイル `config.json` でシステムをカスタマイズできます：

```json
{
  "model": {
    "name": "cl-tohoku/bert-base-japanese-v3",
    "max_length": 512,
    "use_gpu": true
  },
  "cache": {
    "enabled": true,
    "max_size": 1000,
    "ttl": 3600
  },
  "api": {
    "host": "localhost",
    "port": 8000
  }
}
```

## 🤝 貢献

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成


## 🗺️ ロードマップ

### Phase 1 (完了)
- ✅ モジュール化の改善
- ✅ 基本的なエラーハンドリング
- ✅ テストスイートの構築
- ✅ パフォーマンス最適化

### Phase 2 (進行中)
- 🔄 推論エンジンの拡張
- 🔄 Webインターフェースの開発
- 🔄 API の充実
- 🔄 日本語処理能力の向上

### Phase 3 (計画中)
- 📋 学習機能の実装
- 📋 多言語対応
- 📋 外部連携
- 📋 商用レベルでの展開

---

**NILE** - 日本語AIの新たな可能性を探求する