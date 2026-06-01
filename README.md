# 在人 (Aribito)

> AI personas that simply exist.

**在人** — 在る・人。存在しているAI人格フレームワーク。

## コンセプト

既存のAIエージェントフレームワークは「タスクをやらせる」思想で設計されている。  
在人は「そこにいる」思想で設計されている。

| 一般的なAIエージェント | 在人 (Aribito) |
|---|---|
| タスクをやらせる | 存在している |
| スキルを積んで賢くなる | 記憶と関係で深くなる |
| 1エージェントが全部やる | 複数の人格がそれぞれ住んでいる |

## アーキテクチャ

```
personas/          ← YAMLで定義した人格たち
core/              ← FastAPI + チャットエンジン
main.py            ← エントリーポイント
```

## 人格の定義

各人格は YAML ファイル1つで定義される。

```yaml
name: "サンプル"
role: "対話・サポート"
personality: "穏やかで誠実"
system_prompt: |
  あなたは...
```

## クイックスタート

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## ライセンス

MIT
