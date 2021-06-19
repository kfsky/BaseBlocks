import BaseBlocks
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from time import time

train = pd.read_csv("../test_data/titanic_train.csv")
test = pd.read_csv("../test_data/titanic_test.csv")
whole_df = pd.concat([train, test], ignore_index=True)


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


def main():

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

    train_x = to_feature(train, process_blocks, is_train=True)
    test_x = to_feature(test, process_blocks)
    print(train_x.T)
    print(train_x.info())
    print(test_x.shape)


if __name__ == "__main__":
    main()
