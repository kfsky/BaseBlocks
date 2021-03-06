import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import nltk
import texthero as hero
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


# BaseBlock
class BaseBlock(object):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()


# ContinuousBlock
# 連続変数をそのまま使用したいときに使用するBlock
class ContinuousBlock(BaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        return input_df[self.column].copy()


# CountEncoding
class CountEncodingBlock(BaseBlock):
    def __init__(self, column, whole_df: pd.DataFrame):
        self.column = column
        self.whole_df = whole_df

    def transform(self, input_df):
        output_df = pd.DataFrame()
        c = self.column

        vc = self.whole_df[c].value_counts()
        output_df[c] = input_df[c].map(vc)
        return output_df.add_prefix("CE_")


# OneHotEncoding
class OneHotEncodingBlock(BaseBlock):
    def __init__(self, column: str, count_limit: int):
        self.column = column
        self.cats_ = None
        self.count_limit = count_limit

    def fit(self, input_df, y=None):
        vc = input_df[self.column].dropna().value_counts()
        cats = vc[vc > self.count_limit].index
        self.cats_ = cats
        return self.transform(input_df)

    def transform(self, input_df):
        x = pd.Categorical(input_df[self.column], categories=self.cats_)
        output_df = pd.get_dummies(x, dummy_na=False)
        output_df.columns = output_df.columns.tolist()
        return output_df.add_prefix(f'OHE_{self.column}=')


# LabelEncodingBlock
class LabelEncodingBlock(BaseBlock):
    def __init__(self, column: str, whole_df: pd.DataFrame):
        self.column = column
        self.le = LabelEncoder()
        self.whole_df = whole_df

    def fit(self, input_df, y=None):
        self.le.fit(self.whole_df[self.column].fillna("nan"))
        return self.transform(input_df)

    def transform(self, input_df):
        c = self.column
        output_df = pd.DataFrame()
        output_df[c] = self.le.transform(input_df[self.column].fillna("nan")).astype("int")
        return output_df.add_prefix(f'LE_')


# WrapperBlock
# なにか関数をそのまま使用したいときのBlock
class WrapperBlock(BaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)


# 特定カラム同士の四則演算
class ArithmeticOperationBlock(BaseBlock):
    def __init__(self, target_columns1: str, target_columns2: str, operation: str):
        self.target_columns1 = target_columns1
        self.target_columns2 = target_columns2
        self.operation = operation

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df_columns_name = f'{self.target_columns1}{self.operation}{self.target_columns2}'

        if self.operation == "+":
            output_df[output_df_columns_name] = output_df[self.target_columns1] + output_df[self.target_columns2]

        elif self.operation == "-":
            output_df[output_df_columns_name] = output_df[self.target_columns1] - output_df[self.target_columns2]

        elif self.operation == "*":
            output_df[output_df_columns_name] = output_df[self.target_columns1] * output_df[self.target_columns2]

        elif self.operation == "/":
            output_df[output_df_columns_name] = output_df[self.target_columns1] / output_df[self.target_columns2]

        return output_df[output_df_columns_name]


# AggregationBlocks
# 集計用のBlock
class AggregationBlock(BaseBlock):
    def __init__(self, whole_df: pd.DataFrame, key: str, agg_column: str, agg_funcs: ["mean"], fill_na=None):
        self.whole_df = whole_df
        self.key = key
        self.agg_column = agg_column
        self.agg_funcs = agg_funcs
        self.fill_na = fill_na

    def fit(self, input_df):
        if self.fill_na:
            self.whole_df[self.agg_column] = self.whole_df[self.agg_column].fillna(fill_na)

        self.gp_df = self.whole_df.groupby(self.key).agg({self.agg_column: self.agg_funcs}).reset_index()
        column_names = [ f'GP_{self.agg_column}@{self.key}_{agg_func}' for agg_func in self.agg_funcs]
        self.gp_df.columns = [self.key] + column_names
        output_df = pd.merge(input_df[self.key], self.gp_df, on=self.key, how="left").drop(columns=[self.key])
        return output_df

    def transform(self, input_df):
        output_df = pd.merge(input_df[self.key], self.gp_df, on=self.key, how="left").drop(columns=[self.key])
        return output_df


# BinCountBlock
# 連続変数をbin化するBlock
class BinCountBlock(BaseBlock):
    def __init__(self, column: str, bins: int = 10):
        self.bins = bins
        self.column = column

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df_columns_name = f'{self.column}_Bins={self.bins}'
        output_df[output_df_columns_name] = pd.qcut(input_df[self.column], self.bins, labels=False)
        return output_df


# StringLengthBlock
# 文字列の長さを集計するBlock
class StringLengthBlock(BaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.column] = input_df[self.column].str.len()
        return output_df.add_prefix('StringLength_')


# TargetEncodingBlock
# cvはlistの状態
class TargetEncodingBlock(BaseBlock):
    def __init__(self, use_columns, cv):
        super(TargetEncodingBlock, self).__init__()

        self.mapping_df_ = None
        self.use_columns = use_columns
        self.cv = list(cv)
        self.n_fold = len(cv)

    def create_mapping(self, input_df, y):
        self.mapping_df_ = {}
        self.y_mean_ = np.mean(y)

        output_df = pd.DataFrame()
        target = pd.Series(y)

        for col_name in self.use_columns:
            keys = input_df[col_name].unique()
            X = input_df[col_name]

            oof = np.zeros_like(X, dtype=np.float)

            for idx_train, idx_valid in self.cv:
                _df = target[idx_train].groupby(X[idx_train]).mean()
                _df = _df.reindex(keys)
                _df = _df.fillna(_df.mean())
                oof[idx_valid] = input_df[col_name][idx_valid].map(_df.to_dict())

            output_df[col_name] = oof

            self.mapping_df_[col_name] = target.groupby(X).mean()

        return output_df

    def fit(self, input_df:pd.DataFrame,
            y = None, **kwargs)->pd.DataFrame:
        output_df = self.create_mapping(input_df, y=y)
        return output_df.add_prefix("TE_")

    def transform(self, input_df):
        output_df = pd.DataFrame()

        for c in self.use_columns:
            output_df[c] = input_df[c].map(self.mapping_df_[c]).fillna(self.y_mean_)

        return output_df.add_prefix("TE_")


# ShiftBlock
# shiftした値を出力するBlock
class ShiftBlock(BaseBlock):
    def __init__(self, key: str, target_column: str, shift: int):
        self.key = key
        self.target_column = target_column
        self.shift = shift

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.key] = output_df[self.key].fillna("nan")
        output_df[f'Shift{self.shift}_{self.target_column}@{self.key}'] = \
            output_df.groupby(self.key)[self.target_column].transform(lambda x: x.shift(self.shift))

        return output_df[f'Shift{self.shift}_{self.target_column}@{self.key}']


# DiffBlock
# diffした値を出力するBlock
class DiffBlock(BaseBlock):
    def __init__(self, key: str, target_column: str, diff: int):
        self.key = key
        self.target_column = target_column
        self.diff = diff

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.key] = output_df[self.key].fillna("nan")
        output_df[f'Diff{self.diff}_{self.target_column}@{self.key}'] = \
            output_df.groupby(self.key)[self.target_column].transform(lambda x:x.diff(self.diff))

        return output_df[f'Diff{self.diff}_{self.target_column}@{self.key}']


# PivotingBlock
# PCAなど、pivottable作成していく際のBlock
# titanicデータではやりにくいので、別データで確認が必要
class PivotingBlock(BaseBlock):
    def __init__(self, idx, col, val, decomposer=PCA(n_components=4), name=""):
        """
        :param idx: index of pivot table
        :param col: columns of pivot table
        :param val: aggregated feature
        :return: DataFrame(columns=col, index=idx)
        """
        self.idx = idx
        self.col = col
        self.val = val
        self.decomposer = decomposer
        self.name = name
        self.df = None

    def fit(self, input_df, y=None):
        _df = input_df.astype(str).pivot_table(
            index=self.idx,
            columns=self.col,
            values=self.val,
            aggfunc='count',
        ).reset_index()

        idx = _df[self.idx]
        _df.drop(self.idx, axis=1, inplace=True)
        _df = _df.div(_df.sum(axis=1), axis=0).fillna(0)

        if self.decomposer is not None:
            self.df = pd.DataFrame(self.decomposer.fit_transform(_df))
            self.df.columns = [f"{i:03}" for i in range(self.df.shape[1])]
        else:
            self.df = _df.copy()

        self.df.columns = [f"pivot_{self.idx}_{self.col}{self.name}:{s}" for s in self.df.columns]
        self.df[self.idx] = idx

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.idx]], self.df, on=self.idx, how="left").drop(self.idx, axis=1)
        return output_df


# text前処理
def text_normalization(text):
    # 英語、オランダ語をstopwordとして指定
    custom_stopwords = nltk.corpus.stopwords.words("dutch") + nltk.corpus.stopwords.words("english")

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x


# TfidfBlock
# 自然言語系のカラムに対しての処理
class TfidfBlock(BaseBlock):
    """tfidf x SVD による圧縮を行なう block"""

    def __init__(self, column: str):
        """
        args:
            column: str
                変換対象のカラム名
        """
        self.column = column

    def preprocess(self, input_df):
        x = text_normalization(input_df[self.column])
        return x

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す.
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = self.preprocess(input_df)
        self.pileline_ = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svd', TruncatedSVD(n_components=50, random_state=1234)),
        ])

        self.pileline_.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pileline_.transform(text)

        output_df = pd.DataFrame(z)
        return output_df.add_prefix(f'{self.column}_tfidf_')


# Str2Str2OneHotEncoding
# 2つの特徴量を組み合わせて、それをOneHotEncodeする
class Str2Str2OneHotEncodingBlock(BaseBlock):
    def __init__(self, column: str, base_column: str, count_limit: int):
        self.column = column
        self.base_column = base_column
        self.cats_ = None
        self.count_limit = count_limit

    def fit(self, input_df, y=None):
        _df = input_df.copy()
        _df[self.base_column + "_" + self.column] = _df[self.base_column] + "_" + _df[self.column]
        vc = _df[self.base_column + "_" + self.column].dropna().value_counts()
        cats = vc[vc > self.count_limit].index
        self.cats_ = cats
        return self.transform(input_df)

    def transform(self, input_df):
        _df = input_df.copy()
        _df[self.base_column + "_" + self.column] = _df[self.base_column] + "_" + _df[self.column]
        x = pd.Categorical(_df[self.base_column + "_" + self.column], categories=self.cats_)
        output_df = pd.get_dummies(x, dummy_na=False)
        output_df.columns = output_df.columns.tolist()
        return output_df.add_prefix(f'OHE_{self.base_column}@{self.column}=')


# Str2Str2LabelEncoding
# 2つの特徴量を組み合わせて、それをLabelEncodeする
class Str2Str2LabelEncodingBlock(BaseBlock):
    def __init__(self, column: str, base_column: str, whole_df: pd.DataFrame):
        self.column = column
        self.base_column = base_column
        self.le = LabelEncoder()
        self.whole_df = whole_df

    def fit(self, input_df, y=None):
        _df = self.whole_df.copy()
        _df[self.base_column + "_" + self.column] = _df[self.base_column] + "_" + _df[self.column]
        self.le.fit(_df[self.base_column + "_" + self.column].fillna("nan"))
        return self.transform(input_df)

    def transform(self, input_df):
        c = self.base_column + "_" + self.column
        _df = input_df.copy()
        _df[self.base_column + "_" + self.column] = _df[self.base_column] + "_" + _df[self.column]
        output_df = pd.DataFrame()
        output_df[c] = self.le.transform(_df[self.base_column + "_" + self.column].fillna("nan")).astype("int")
        return output_df.add_prefix(f'LE_')


# Str2Str2LabelEncoding
# 2つの特徴量を組み合わせて、それをLabelEncodeする
class Str2Str2CountEncodingBlock(BaseBlock):
    def __init__(self, column: str, base_column: str, whole_df: pd.DataFrame):
        self.column = column
        self.base_column = base_column
        self.whole_df = whole_df

    def transform(self, input_df):
        _df = self.whole_df.copy()
        _df[self.base_column + "_" + self.column] = _df[self.base_column] + "_" + _df[self.column]

        c = self.base_column + "_" + self.column
        vc = _df[c].value_counts()

        _input_df = input_df.copy()
        _input_df[self.base_column + "_" + self.column] = _input_df[self.base_column] + "_" + _input_df[self.column]

        output_df = pd.DataFrame()
        output_df[c] = _input_df[c].map(vc)

        return output_df.add_prefix("CE_")

