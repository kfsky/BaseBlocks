import pandas as pd
import numpy as np


# BaseBlock
class BaseBlock(object):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()


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


# WrapperBlock
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



