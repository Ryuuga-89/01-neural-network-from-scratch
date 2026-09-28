# Neural Networks from Scratch

NumPyを用いて、ニューラルネットワークの主要な構成要素をゼロから実装する学習プロジェクト。

本プロジェクトの目的は、ニューラルネットワークを高水準ライブラリから利用することではなく、各コンポーネントの内部で行われている計算を理解し、自ら実装することで、ニューラルネットワークの動作原理をコードレベルで理解することである。

特に、

- 順伝播
- 逆伝播
- パラメータ更新
- CNN系レイヤ
- 正規化

といった処理について、数式上の理解と実装を対応付けることを重視している。

本プロジェクトは『ゼロから作るDeep Learning』第1巻の学習テーマを参考にしているが、書籍および公式リポジトリのコードは使用していない。

## Goals

本プロジェクトでは、ニューラルネットワークを構成する主要なコンポーネントについて、

1. 内部で行われている計算を理解する
2. NumPyを用いて自力で実装する
3. forward / backwardの挙動をテストによって検証する
4. 各コンポーネントを組み合わせてMLPおよびCNNを構築する
5. 実際に学習可能であることを小規模な実験によって確認する

ことを目標とする。

ライブラリとしての機能性や実用的な学習性能を追求することではなく、ニューラルネットワークの各構成要素を理解することを主眼としている。

## Implementation Policy

数値計算の基盤としてNumPyを使用する。

NumPyには以下のような低水準の数値処理を任せる。

- 配列の生成・保持
- 行列積
- element-wise演算
- reduction
- reshape / transpose
- 乱数生成
- 基本的な線形代数演算

一方、ニューラルネットワーク固有の処理は自力で実装する。

PyTorch、TensorFlow、JAX等のDeep Learning Frameworkは、ニューラルネットワーク本体の実装および学習には使用しない。

コアとなるニューラルネットワーク実装は、自身でアルゴリズムを理解した上で実装している。

テストについては、実装の検証を目的としてAIを利用したテストケースの設計・レビューも行っている。

## Implemented Components

### Layers

[`src/neural_networks/layers/`](src/neural_networks/layers)

#### Basic Layers

[`basic_layers.py`](src/neural_networks/layers/basic_layers.py)

- `Linear`
- `ReLU`
- `Sigmoid`
- `Flatten`

各layerは基本的に、

```python
forward_propagation(x)
backward_propagation(dout)
```

を持ち、順伝播と逆伝播の双方を実装している。

#### Convolutional Layers

[`convolutional_layers.py`](src/neural_networks/layers/convolutional_layers.py)

- `Conv2D`
- `MaxPooling2D`

`Conv2D`ではstrideやpaddingを含む二次元畳み込みを扱う。

`MaxPooling2D`ではpooling window内の最大値の選択と、それに対応する逆伝播を実装している。

#### Other Layers

[`other_layers.py`](src/neural_networks/layers/other_layers.py)

- `Dropout`
- `BatchNormalization`
- `Sequential`

`Sequential`によって複数のlayerを組み合わせ、MLPやCNNを構築できる。

例：

```python
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
)
```

### Loss Functions

[`src/neural_networks/loss_functions/loss_functions.py`](src/neural_networks/loss_functions/loss_functions.py)

以下の損失関数を実装している。

- `MeanSquaredErrorLoss`
- `CrossEntropyLoss`
- `BCEWithLogitsLoss`

forwardで損失を計算するとともに、backwardで入力に対する勾配を計算する。

数値計算上問題となる箇所については、可能な範囲で数値安定性を考慮して実装している。

### Optimizers

[`src/neural_networks/optimizers/optimizers.py`](src/neural_networks/optimizers/optimizers.py)

以下の最適化アルゴリズムを実装している。

- `SGD`
- `Momentum`
- `AdaGrad`
- `Adam`

各optimizerでは、modelが保持するparameterおよびgradientを用いてparameter updateを行う。

特にAdamについては、

- first moment
- second moment
- bias correction

を含む更新則を実装している。

### DataLoader

[`src/neural_networks/data_loader/data_loader.py`](src/neural_networks/data_loader/data_loader.py)

- `DataLoader`

datasetからmini-batchを生成し、ニューラルネットワークの学習に利用する。

### Trainer

[`src/neural_networks/trainer/trainer.py`](src/neural_networks/trainer/trainer.py)

- `Trainer`

model、loss function、optimizer、DataLoaderを組み合わせて学習を実行するための処理をまとめている。

基本的なtraining loopをニューラルネットワーク本体の実装から分離するためのコンポーネントである。

## Verification

本プロジェクトでは、単にdataset上でaccuracyが得られることだけを実装の正しさの根拠とはしない。

各コンポーネントについて、[`tests/`](tests) 以下にテストを作成している。

テストでは主に以下を確認する。

### Forward Tests

既知の入力に対して、順伝播の結果が期待値と一致することを確認する。

特に単純な入力については、手計算または独立した参照計算と比較する。

### Backward Tests

各layerについて、逆伝播によって得られるgradientが期待する値と一致することを確認する。

parameterを持つlayerでは、

- input gradient
- weight gradient
- bias gradient

などをそれぞれ検証する。

### Shape Tests

入力shapeに対して、forward / backwardの双方で期待するshapeが維持されることを確認する。

特に、

- batch dimension
- multi-channel input
- convolution kernel
- stride
- padding
- pooling

などを含むケースを検証する。

### Edge Cases

通常ケースだけでなく、実装上問題になりやすい境界条件についてもテストする。

例えばconvolution / poolingでは、

- rectangular kernel
- different strides
- different padding modes
- overlapping regions
- multi-channel inputs

などについて検証する。

### Reference Comparison

必要に応じて、PyTorch等の既存実装をテスト上の参照値として利用する。

これはニューラルネットワーク本体の処理をframeworkに委譲するためではなく、自作実装の数値的な正しさを検証するためにのみ使用する。

## Experiments

本プロジェクトでは、optimizerやarchitectureの性能比較を主目的とした大規模な実験は行わない。

実験の目的は、

> 実装したコンポーネントを組み合わせたニューラルネットワークが、実際に正常に学習できるか

を確認することである。

実験コードは [`experiments/`](experiments) 以下に配置する。

### MLP Overfitting Test

実装した、

- `Linear`
- activation function
- loss function
- optimizer
- `DataLoader`
- `Trainer`

などを組み合わせてMLPを構築する。

小規模なdatasetまたはdataset subsetについて学習を行い、十分なmodel capacityを与えたときにtraining dataへoverfitできることを確認する。

これによって、

```text
DataLoader
    ↓
Model
    ↓
Loss
    ↓
Backward
    ↓
Optimizer
```

というMLPのtraining pipeline全体が正常に動作していることを確認する。

### CNN Overfitting Test

実装した、

- `Conv2D`
- activation function
- `MaxPooling2D`
- `Flatten`
- `Linear`

などを組み合わせてCNNを構築する。

MLPと同様に小規模なdatasetについて学習し、training dataへ十分にoverfitできることを確認する。

これによって、convolutionおよびpoolingを含むCNNのforward / backward処理が、training pipeline全体として正常に機能していることを確認する。

## Project Structure

主要なディレクトリ構成は以下の通り。

```text
.
├── src/
│   └── neural_networks/
│       ├── layers/
|       |   ├── abstract_layers.py
│       │   ├── basic_layers.py
│       │   ├── convolutional_layers.py
│       │   └── other_layers.py
│       ├── loss_functions/
│       │   └── loss_functions.py
│       ├── optimizers/
│       │   └── optimizers.py
│       ├── data_loader/
│       │   └── data_loader.py
│       └── trainer/
│           └── trainer.py
├── tests/
├── experiments/
├── docs/
│   ├── implementation_note/
|   └── report/
├── README.md
├── pyproject.toml
└── uv.lock
```

`src/`にはニューラルネットワーク本体の実装を配置する。

`tests/`には各コンポーネントの単体テストおよび結合的なテストを配置する。

`experiments/`にはMLPおよびCNNが実際に学習可能であることを確認するための小規模な実験を配置する。

実装中に得られた設計上の知見や判断については、必要に応じて`docs/implementation_philosophy/`以下に記録する。

## Usage

依存関係のインストール：

```bash
uv sync
```

テストの実行：

```bash
uv run pytest
```

MLP / CNNの学習実験については、`experiments/`以下のスクリプトから実行する。

例としては

```bash
uv run python -m experiments.03_CNN-MulticlassClassification.main
```

## Scope

本プロジェクトで重視するのは、

- ニューラルネットワークの各コンポーネントの理解
- forward / backwardの実装
- 数値計算の詳細の理解
- テストによる自作実装の検証
- 各コンポーネントを組み合わせたMLP / CNNの構築

である。

一方、以下は本プロジェクトの主要な対象とはしない。

- optimizer間の詳細なbenchmark
- hyperparameter tuning
- MLPとCNNの性能比較
- 大規模datasetでのtraining
- GPUによる高速化
- automatic differentiation
- production向けDeep Learning Frameworkとしての機能性
- 詳細な実験レポートの作成

これらの機能を追加することよりも、ニューラルネットワーク内部の計算を理解し、それを自力で実装することを優先する。

## Completion Criteria

本プロジェクトでは、以下を主要な完了条件とする。

- [x] `Linear`を実装した
- [x] `ReLU`を実装した
- [x] `Sigmoid`を実装した
- [x] `Flatten`を実装した
- [x] `Conv2D`を実装した
- [x] `MaxPooling2D`を実装した
- [x] `Dropout`を実装した
- [x] `BatchNormalization`を実装した
- [x] `Sequential`を実装した
- [x] `MeanSquaredErrorLoss`を実装した
- [x] `CrossEntropyLoss`を実装した
- [x] `BCEWithLogitsLoss`を実装した
- [x] `SGD`を実装した
- [x] `Momentum`を実装した
- [x] `AdaGrad`を実装した
- [x] `Adam`を実装した
- [x] `DataLoader`を実装した
- [x] `Trainer`を実装した
- [x] 各主要コンポーネントに対するテストを作成した
- [x] MLPが小規模datasetへoverfitできることを確認する
- [x] CNNが小規模datasetへoverfitできることを確認する
- [x] 全自動テストが成功することを確認する

## Status

NNの全コーンポーネントの実装、各種実験の実装を完了

## References

- 斎藤康毅『ゼロから作るDeep Learning ― Pythonで学ぶディープラーニングの理論と実装』
- NumPy Documentation
- 検証時に参照したPyTorch等の公式ドキュメント