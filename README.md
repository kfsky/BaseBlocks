# BaseBlocks
コンペの特徴量作成を行っていく際に、各処理をBlock化。  
特徴量生成の際に、作成した特徴量の処理がわかりやすくなる・他特徴量への影響をなくすなどの目的で作っています。  

## フォルダ構成
BaseBlocks.pyにBlocksを記載しています。

.  
├── BaseBlocks  
│   ├── BaseBlocks.py  
│   └── tests.py（動作確認用ファイル）  
└── test_data  
    ├── titanic_train.csv  
    └── titanic_test.csv  
  
テストにはtitanicデータを使用しています。  
https://www.kaggle.com/c/titanic/data

## Blocks
以下のBlocksを実装

* ContinuousBlock
* CountEncodingBlock
* OneHotEncodingBlock
* LabelEncodingBlock
* WrapperBlock
* ArithmeticOperationBlock
* AggregationBlock
* BinCountBlock
* StringLengthBlock
* TargetEncodingBlock
* ShiftBlock
* DiffBlock
* PivotingBlock
* TfidfBlock
* Str2Str2OneHotEncodingBlock
* Str2Str2LabelEncodingBlock
* Str2Str2CountEncodingBlock

## 実装方法

test.pyに記載。process_blocks(list)に必要なBlocksをいれて、to_featureで実行
```python
@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)

def get_function(block, is_train):
    s = mapping = {
        True: 'fit',
        False: 'transform'
    }.get(is_train)
    return getattr(block, s)

# Blocksを実行する
def to_feature(input_df,
               blocks,
               is_train=False):
    out_df = pd.DataFrame()

    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)

        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)

        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)

    return out_df
```

process_blocksの例（必要なBlockをリストに格納）  
複数カラムでの実行も可能
```python
process_blocks = [
        *[BaseBlocks.ContinuousBlock(c) for c in ["Age"]],
        BaseBlocks.BinCountBlock("Age", bins=8),
        *[BaseBlocks.OneHotEncodingBlock(c, count_limit=10) for c in ['Sex']],
        *[BaseBlocks.CountEncodingBlock(c, whole_df=whole_df) for c in ['Sex']],
        *[BaseBlocks.LabelEncodingBlock(c, whole_df=whole_df) for c in ['Sex']],
        *[BaseBlocks.DiffBlock(key=c, target_column="Age", diff=1) for c in ["Sex"]],
        *[BaseBlocks.StringLengthBlock(c) for c in ["Ticket"]],
        *[BaseBlocks.ArithmeticOperationBlock("Fare", "Age", "/")],
        *[BaseBlocks.AggregationBlock(whole_df=whole_df, key=c, agg_column="Age",
                                      agg_funcs=["mean"], fill_na=0) for c in ["Sex"]]
    ]
```

train_data, test_dataそれぞれで特徴量生成。（Blockによってはwhole_dfが必要）
```python
train = pd.read_csv("../test_data/titanic_train.csv")
test = pd.read_csv("../test_data/titanic_test.csv")
whole_df = pd.concat([train, test], ignore_index=True)

train_x = to_feature(train, process_blocks, is_train=True)
test_x = to_feature(test, process_blocks)
```

## install
以下でinstallして使用可能。（テストできてないところがあるので、
Global以外で行う方がいいです。
```
python -m venv .venv
source .venv/bin/activate
```

install完了後は以下のようにimportして実装
```python
from BaseBlocks import BaseBlocks
```

## To Do
* test環境？など
* modelなどの関数実装→これは他のリポジトリでやっていくかも
