# Neural Networks from Scratch

NumPyを用いて、ニューラルネットワークの主要な構成要素をゼロから実装するプロジェクト。

本プロジェクトの目的は、ニューラルネットワークを高水準ライブラリから利用することではなく、forward propagation、backpropagation、optimization、convolutionなどの計算を自ら実装し、その動作を数値的に検証することで、深層学習の基礎原理を理解することである。

本プロジェクトは『ゼロから作るDeep Learning』第1巻の学習テーマに対応するが、書籍および公式リポジトリのコードは使用しない。

## Goals

本プロジェクトでは、以下の能力を身につけることを目標とする。

* ニューラルネットワークのforward propagationを数式とコードの両方で説明できる
* backpropagationを導出し、自力で実装できる
* 各パラメータに対する勾配を数値的に検証できる
* SGDをはじめとする主要なoptimizerを実装できる
* weight initialization、Batch Normalization、Dropoutが学習に与える影響を説明できる
* convolutionおよびpoolingを実装し、CNNを構築できる
* 学習・評価・比較実験を再現可能な形で実行できる
* 実装が正しいことをテストと数値実験によって確認できる

## Rules

本リポジトリには、本学習プロジェクト共通の `GLOBAL-RULES.md` を適用する。

特に、以下を遵守する。

* 書籍・外部サイト・既存リポジトリからコードを転記しない
* AIエージェントおよび生成AIを実装・設計・デバッグに使用しない
* `.ipynb` を使用しない
* 中核アルゴリズムを実装済みの高水準ライブラリに委譲しない
* 全ての主要実装に対して適切なテストを作成する
* 実験はCLIから再現可能にする
* 乱数を明示的に制御する

## Scope

### Numerical Backend

数値計算の基盤としてNumPyを使用する。

NumPyには以下のような低水準の数値処理を任せる。

* 配列の生成・保持
* 行列積
* element-wise演算
* reduction
* reshape / transpose
* 乱数生成
* 基本的な線形代数演算

一方、ニューラルネットワーク固有の処理は自分で実装する。

PyTorch、TensorFlow、JAX等のdeep learning frameworkは、モデル実装・学習には使用しない。

## Required Implementation

### Basic Layers

以下のlayerを実装する。

* `Linear`
* `ReLU`
* `Sigmoid`
* `Flatten`

各layerは少なくとも以下の処理を持つ。

```python
forward(x)
backward(grad_output)
```

### Loss Functions

以下を実装する。

* Mean Squared Error
* Softmax
* Cross Entropy Loss
* Softmax Cross Entropy Loss

数値安定性を考慮した実装とする。

### Optimization

以下のoptimizerを実装する。

* SGD
* Momentum
* AdaGrad
* Adam

optimizerとmodel parameterの管理を分離する。

### Weight Initialization

以下の初期化方式を実装する。

* Standard random initialization
* Xavier initialization
* He initialization

### Regularization and Training Techniques

以下を実装する。

* Dropout
* Batch Normalization

training modeとevaluation modeの挙動を適切に分離する。

### Convolutional Neural Networks

以下を実装する。

* `Conv2D`
* `MaxPool2D`

少なくともstrideおよびpaddingを扱えるようにする。

これらを利用してCNNを構築する。

### Model Abstraction

layerを組み合わせてネットワークを構築できる最小限の仕組みを実装する。

例：

```python
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
)
```

ただし、API設計は実装過程で必要に応じて変更してよい。

### Training Infrastructure

以下を実装する。

* mini-batch training
* training loop
* validation loop
* metric calculation
* parameter update
* checkpoint saving/loading
* reproducible seeding

## Verification

本プロジェクトでは、モデルのaccuracyだけを実装の正しさの根拠としない。

### Gradient Checking

backpropagationによって得られた解析的勾配を、有限差分による数値微分と比較する。

中央差分

$$
\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}
$$

を利用し、relative errorを計算する。

少なくとも以下についてgradient checkを実施する。

* `Linear`
* activation functions
* loss functions
* `BatchNorm`
* `Conv2D`

必要に応じてその他のlayerについても実施する。

### Known-value Tests

手計算可能な小さな入力を用意し、forwardおよびbackwardの結果が期待値と一致することを確認する。

### Shape Tests

各layerについて、入力shapeに対して期待する出力shapeおよびgradient shapeが得られることを確認する。

特に以下を重点的に確認する。

* batch dimension
* convolution
* padding
* stride
* pooling

### Tiny Dataset Overfitting

少数のtraining samplesのみを使用し、十分なmodel capacityを持つネットワークがほぼ完全にoverfitできることを確認する。

この実験によって、training pipeline全体が正常に動作していることを検証する。

### Serialization Test

modelを保存・再読み込みした後、同じ入力に対して同じ出力が得られることを確認する。

## Experiments

### Experiment 1: Optimizer Comparison

以下を同一条件で比較する。

* SGD
* Momentum
* AdaGrad
* Adam

比較対象：

* training loss
* validation loss
* convergence speed
* final validation accuracy

### Experiment 2: Weight Initialization

以下を比較する。

* standard random initialization
* Xavier initialization
* He initialization

各層におけるactivationおよびgradientの分布も必要に応じて計測する。

### Experiment 3: Batch Normalization

Batch Normalizationの有無を比較する。

評価する項目：

* convergence
* training stability
* validation performance
* activation distribution

### Experiment 4: Dropout

Dropoutの有無、および複数のdropout probabilityを比較する。

training performanceとgeneralizationの関係を分析する。

### Experiment 5: MLP vs CNN

同一の画像分類datasetについて、

* Multi-Layer Perceptron
* Convolutional Neural Network

を比較する。

単純な最終accuracyだけでなく、parameter数、学習時間、学習曲線も比較する。

## Dataset

最終評価には、MNISTまたはFashion-MNIST相当の画像分類datasetを使用する。

dataset取得および前処理処理は、中核となるニューラルネットワーク実装とは分離する。

データ分割はtraining / validation / testを明確に区別し、test setをモデル選択に使用しない。

## Reproducibility

全ての実験で乱数seedを明示的に指定可能にする。

例：

```bash
uv run python -m experiments.compare_optimizers --seed 42
```

複数seedを使用する実験では、使用したseed集合を実験設定として保存する。

実験結果とグラフはコードから生成し、手作業による数値変更を行わない。

## Project Structure

想定するディレクトリ構成：

```text
.
├── src/
│   └── neural_networks/
│       ├── layers/
│       ├── losses/
│       ├── optimizers/
│       ├── initializers/
│       ├── models/
│       ├── training/
│       └── utils/
├── tests/
├── experiments/
├── scripts/
├── configs/
├── assets/
│   └── figures/
├── reports/
│   └── technical_report.md
├── GLOBAL-RULES.md
├── README.md
├── pyproject.toml
└── uv.lock
```

ディレクトリ構成は実装の進行に応じて変更する可能性がある。

## CLI

最終的に主要な処理をCLIから実行できる状態にする。

想定例：

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Train MLP
uv run python -m experiments.train_mlp

# Train CNN
uv run python -m experiments.train_cnn

# Compare optimizers
uv run python -m experiments.compare_optimizers

# Compare initialization methods
uv run python -m experiments.compare_initialization

# Evaluate Batch Normalization
uv run python -m experiments.batchnorm

# Evaluate Dropout
uv run python -m experiments.dropout

# Compare MLP and CNN
uv run python -m experiments.compare_architectures
```

具体的なCLIは実装後に確定する。

## Technical Report

`reports/technical_report.md` に、本プロジェクトで学習・実装・検証した内容をまとめる。

少なくとも以下を含める。

1. Neural Networks
2. Forward Propagation
3. Loss Functions
4. Backpropagation
5. Gradient Checking
6. Optimization Algorithms
7. Weight Initialization
8. Batch Normalization
9. Dropout
10. Convolutional Neural Networks
11. Verification Methodology
12. Experimental Setup
13. Results
14. Discussion
15. Limitations

READMEはプロジェクト全体の概要と主要成果を示し、詳細な理論・実験考察はTechnical Reportに記載する。

## Completion Criteria

以下を全て満たした時点で本プロジェクトを完了とする。

* [ ] 全必須layerを実装した
* [ ] 全必須loss functionを実装した
* [ ] SGD / Momentum / AdaGrad / Adamを実装した
* [ ] Xavier / He initializationを実装した
* [ ] Batch Normalizationを実装した
* [ ] Dropoutを実装した
* [ ] Conv2Dを実装した
* [ ] MaxPool2Dを実装した
* [ ] MLPを学習できる
* [ ] CNNを学習できる
* [ ] 必須gradient checkを通過する
* [ ] 全自動テストが成功する
* [ ] Tiny Dataset Overfitting Testを通過する
* [ ] Optimizer比較実験を完了した
* [ ] Weight Initialization比較実験を完了した
* [ ] Batch Normalization比較実験を完了した
* [ ] Dropout比較実験を完了した
* [ ] MLP vs CNN比較実験を完了した
* [ ] 全主要実験をCLIから再現できる
* [ ] Technical Reportを完成させた
* [ ] READMEに最終結果を掲載した
* [ ] `GLOBAL-RULES.md` の全要件を満たしている

全条件を満たした時点で `v1.0.0` をリリースする。

## References

* 斎藤康毅『ゼロから作るDeep Learning ― Pythonで学ぶディープラーニングの理論と実装』
* 使用した論文・技術資料・公式ドキュメントは、実装およびレポート作成の進行に応じて追記する

## Status

**Planning / Not yet completed**

本READMEは実装前の仕様を兼ねている。

実装・実験の進行に伴い、実際の結果、グラフ、設計上の判断、制約等を追記する。
