# BaseBlocks
BaseBlockを書いて特徴量を作成していく

## フォルダ構成
BaseBlocks.pyにBlocksを記載しています。

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

## To Do
* PCAなどのBlock作成
* str同士の組み合わせを行い、LE, OHEなどを行えるBlock
