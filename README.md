# zenn-content

[IntereStat](https://www.interestat.jp)（統計を「前提を置いて判断する道具」として扱う学習サイト）のコラムの実験部分を、コード付きで公開しています。

- 記事本体は [Zenn](https://zenn.dev/haruguri) にあります
- 図の再現スクリプトは `scripts/` にあります（numpy と matplotlib のみで動きます）
- 式を減らして具体例で書いた版は IntereStat のコラムにあります

## 記事一覧

| 記事 | 対応するコラム | スクリプト |
|---|---|---|
| [独立が成り立たないとき、大数の法則に何が起きるか](Zennの記事URL・公開後に差し替え) | [1,000人の世論調査は、本当に1,000人分か](https://www.interestat.jp/columns/yoron-chousa-taisuu-housoku) | [lln_dependence.py](scripts/lln_dependence.py) |

## 構成

- `articles/` — Zenn の記事（Markdown）
- `images/` — 記事の図
- `scripts/` — 図と表を生成するスクリプト。乱数のシードを固定してあるので、実行すると記事と同じ図が出ます
