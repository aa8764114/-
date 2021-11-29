import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

pd.options.display.max_columns = None  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # 取消科学计数法

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')


def get_kind(x: pd.Series, diff_limit: int = 10):
    x = x.astype('str')
    x = x.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    x.dropna(inplace=True)
    if x.nunique() > diff_limit:
        kind = 'numeric'
    else:
        kind = 'categorical'
    return kind


def check_data_y(X):
    """
    检查数据结构，数据预测变量为 0,1，并以“y”命名
    """
    if 'y' not in X.columns:
        logging.error('未检测到"y"变量，请将预测变量命名改为"y"')


def get_cate_num(D, c_list, n_list):
    if n_list is None and c_list is None:
        n_list = D._get_numeric_data().columns
        c_list = list(set(D.columns).difference(set(n_list)))
    elif n_list is not None and c_list is not None:
        pass
    else:
        if c_list is None:
            c_list = list(set(D.columns).difference(set(n_list)))
        else:
            n_list = list(set(D.columns).difference(set(c_list)))
    return c_list, n_list


class define_column_type(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 cate_type: str = 'object',
                 num_type: str = 'float',
                 diff_num: int = 10):
        self.num_list = num_list
        self.cate_list = cate_list
        self.cate_type = cate_type
        self.num_type = num_type
        self.diff_num = diff_num

    def fit(self, X, y=None):
        if self.num_list is None and self.cate_list is None:
            self.num_list, self.cate_list = [], []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)
                else:
                    self.cate_list.append(col)
        elif self.num_list is not None and self.cate_list is not None:
            pass
        else:
            if self.cate_list is None:
                self.cate_list = list(set(X.columns).difference(set(self.num_list)))
            else:
                self.num_list = list(set(X.columns).difference(set(self.cate_list)))
        logging.info('Fit Columns Type Already Success!!!')
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cate_list:
            X[col] = X[col].astype(self.cate_type)
        for col in self.num_list:
            X[col] = X[col].astype(self.num_type)
        logging.info('Transform Columns Type Success!!!')
        return X


class wrong_value_fillna(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 wrong_value: list = None,
                 diff_num: int = 10,
                 reset_coltype: bool = True,
                 cate_type: str = 'object',
                 num_type: str = 'float'):
        self.num_list = num_list
        self.cate_list = cate_list
        self.wrong_value = wrong_value
        self.diff_num = diff_num
        self.reset_coltype = reset_coltype
        self.cate_type = cate_type
        self.num_type = num_type

    def fit(self, X, y=None):
        X = X.copy()
        '''get num_list and cate_list'''
        if self.num_list is None:
            self.num_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)
        if self.cate_list is None:
            self.cate_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'categorical':
                    self.cate_list.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        X.replace(self.wrong_value, np.nan, inplace=True)
        if self.reset_coltype:
            for col in self.num_list:
                X[col] = X[col].astype(self.num_type)
            for col in self.cate_list:
                X[col] = X[col].astype(self.cate_type)
        logging.info('Wrong Values has Filled as np.nan!!!')
        return X


class base_fill(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 diff_num: int = 10,
                 cate_method: str = 'self_value',
                 cate_value: str = 'Unknown',
                 num_method: str = 'self_value',
                 num_value: float = 0,
                 self_fill_dict: dict = None
                 ):
        self.num_list = num_list
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.cate_method = cate_method
        self.cate_value = cate_value
        self.num_method = num_method
        self.num_value = num_value
        self.self_fill_dict = self_fill_dict

    def fit(self, X, y=None):
        from tqdm import tqdm
        X = X.copy()
        if self.self_fill_dict is not None:
            return self
        else:
            '''get num_list and cate_list'''
            self.cate_list, self.num_list = get_cate_num(X, self.cate_list, self.num_list)
            '''fill nan'''
            self.self_fill_dict = {}
            for col in self.num_list:
                if self.num_method == 'self_value':
                    self.self_fill_dict[col] = self.num_value
                elif self.num_method == 'mean':
                    self.self_fill_dict[col] = X[col].mean()
                elif self.num_method == 'median':
                    self.self_fill_dict[col] = X[col].median()
                else:
                    logging.error('Do not Find this num_method. Please Input "self_value"/"median"/"mean". ')
            for col in self.cate_list:
                if self.cate_method == 'self_value':
                    self.self_fill_dict[col] = self.cate_value
                elif self.cate_method == 'mode':
                    self.self_fill_dict[col] = X[col].mode()[0]
                else:
                    logging.error('Do not Find this cate_method. Please Input "self_value"/"mode". ')

            return self

    def transform(self, X):
        X = X.copy()
        from tqdm import tqdm
        for key in tqdm(self.self_fill_dict):
            X[key] = X[key].fillna(self.self_fill_dict[key])
        logging.info('base_fill for NA Success!!!')
        return X


class fill_default_rate(BaseEstimator, TransformerMixin):

    def __init__(self,
                 columns: list = None,
                 pos_label: int = 1,
                 ):
        self.columns = columns
        self.pos_label = pos_label
        self.rate_dict = defaultdict(defaultdict)

    def fit(self, X, y=None):
        X = X.copy()
        X['y'] = y
        for col in self.columns:
            for fea in list(X[col].unique()):
                self.rate_dict[col][fea] = (X[(X['y'] == self.pos_label) & (X[col] == fea)].shape[0]) / (
                    X[X[col] == fea].shape[0])
        # print(self.rate_dict)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            for fea in list(X[col].unique()):
                X[col].replace(fea, self.rate_dict[col][fea], inplace=True)
        logging.info('fill_default_rate Success!!!')
        return X


class xgb_fill(BaseEstimator, TransformerMixin):

    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 diff_num: int = 10,
                 random_state: int = 0):
        self.num_list = num_list
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.random_state = random_state
        self.xgb_cla_dict = {}
        self.xgb_reg_dict = {}
        self.fit_cla_dict = {}
        self.fit_reg_dict = {}

    def fit(self, X, y=None):
        from tqdm import tqdm
        X = X.copy()
        self.cate_list, self.num_list = get_cate_num(X, self.cate_list, self.num_list)
        for col in tqdm(self.cate_list):
            file = X.copy()
            df = pd.get_dummies(file, columns=[i for i in self.cate_list if i != col],
                                prefix=[i for i in self.cate_list if i != col],
                                dummy_na=True)
            not_null = df.dropna(subset=[col])
            x_ = not_null.drop([col], axis=1)
            x_.sort_index(axis=1, inplace=True)
            y_ = not_null[col]
            xgb_cla = xgb.XGBClassifier(random_state=self.random_state)
            xgb_cla.fit(x_, y_)
            self.xgb_cla_dict[col] = xgb_cla
            self.fit_cla_dict[col] = x_.columns

        for col in tqdm(self.num_list):
            file = X.copy()
            df = pd.get_dummies(file, columns=self.cate_list, dummy_na=True, prefix=self.cate_list)
            not_null = df.dropna(subset=[col])
            x_ = not_null.drop([col], axis=1)
            x_.sort_index(axis=1, inplace=True)
            y_ = not_null[col]
            xgb_reg = xgb.XGBRegressor(random_state=self.random_state, objective='reg:squarederror')
            xgb_reg.fit(x_, y_)
            self.xgb_reg_dict[col] = xgb_reg
            self.fit_reg_dict[col] = x_.columns
        logging.info('xgb_fill Fit Success!!!')
        return self

    def transform(self, X):
        X = X.copy()
        from tqdm import tqdm
        for col in tqdm(self.cate_list):
            file = X.copy()
            if file[col].isnull().any():
                df = pd.get_dummies(file, columns=[i for i in self.cate_list if i != col],
                                    prefix=[i for i in self.cate_list if i != col],
                                    dummy_na=True)
                fit_only = set(self.fit_cla_dict[col]).difference(set(df.columns))
                trans_only = set(df.columns).difference(set(self.fit_cla_dict[col]))
                if len(fit_only) != 0:
                    for item in fit_only:
                        df[item] = np.zeros((df.shape[0],), dtype=np.int)
                else:
                    pass
                if len(trans_only) != 0:
                    for item in trans_only:
                        if item != col:
                            df.drop(item, axis=1, inplace=True)
                        else:
                            pass
                else:
                    pass

                not_null = df.dropna(subset=[col])
                null = df.drop(not_null.index)
                null.drop([col], axis=1, inplace=True)
                null.sort_index(axis=1, inplace=True)
                null[col] = self.xgb_cla_dict[col].predict(null)  # 字段顺序对应不上
                X[col] = pd.concat([null, not_null], axis=0)[col]
            else:
                X[col] = file[col]

        for col in tqdm(self.num_list):
            file = X.copy()
            if file[col].isnull().any():
                df = pd.get_dummies(file, columns=self.cate_list, dummy_na=True, prefix=self.cate_list)
                fit_only = set(self.fit_reg_dict[col]).difference(set(df.columns))
                trans_only = set(df.columns).difference(set(self.fit_reg_dict[col]))
                if len(fit_only) != 0:
                    for item in fit_only:
                        df[item] = np.zeros((df.shape[0],), dtype=np.int)
                else:
                    pass
                if len(trans_only) != 0:
                    for item in trans_only:
                        if item != col:
                            df.drop(item, axis=1, inplace=True)
                        else:
                            pass
                else:
                    pass

                not_null = df.dropna(subset=[col])
                null = df.drop(not_null.index)
                null.drop([col], axis=1, inplace=True)
                null.sort_index(axis=1, inplace=True)
                null[col] = self.xgb_reg_dict[col].predict(null)
                X[col] = pd.concat([null, not_null], axis=0)[col]
            else:
                X[col] = file[col]
        logging.info('xgb_fill for NA Success!!!')
        return X


class rf_fill(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cate_list: list = None,
                 diff_num: int = 10,
                 random_state: int = 0):
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.imputer = None
        self.le_dict = {}
        self.random_state = random_state

    def fit(self, X, y=None):
        from tqdm import tqdm
        X = X.copy()
        self.cate_list, num_list = get_cate_num(X, self.cate_list, None)

        cate_num = []
        for col in tqdm(self.cate_list):
            cate_num.append(list(X.columns).index(col))
            from sklearn.preprocessing import LabelEncoder
            exist_na = False
            if X[col].isnull().any():  # 是否有空值
                X[col].fillna('NaN', inplace=True)
                exist_na = True

            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.le_dict[col] = le

            if exist_na:
                X[col].replace({list(self.le_dict[col].classes_).index('NaN'): np.nan}, inplace=True)

        import sklearn.neighbors._base
        import sys
        sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

        from missingpy import MissForest
        imputer = MissForest()
        self.imputer = imputer.fit(X, cat_vars=cate_num)
        logging.info('rf_fill Fit Success!!!')
        return self

    def transform(self, X):
        X = X.copy()
        from tqdm import tqdm
        cate_num = []
        encoder_dict = {}
        for col in tqdm(self.cate_list):
            cate_num.append(list(X.columns).index(col))
            exist_na = False
            if X[col].isnull().any():  # 是否有空值
                X[col].fillna('NaN', inplace=True)
                exist_na = True
            # fit_only = set(list(self.le.classes_)).difference(list(X[col].unique()))
            trans_only = set(list(X[col].unique())).difference(set(list(self.le_dict[col].classes_)))
            if len(trans_only) != 0:
                X[col] = X[col].replace(list(trans_only), 'NaN')
            else:
                pass

            X[col] = self.le_dict[col].transform(X[col])
            encoder_dict[col] = dict(enumerate(self.le_dict[col].classes_))

            if exist_na:
                X[col].replace({list(self.le_dict[col].classes_).index('NaN'): np.nan}, inplace=True)

        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)

        for col in tqdm(self.cate_list):
            X[col].replace(encoder_dict[col], inplace=True)

        logging.info('rf_fill Success!!!')
        return X


class fix_outlier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list=None,
                 diff_num: int = 10,
                 pmin: float = None,
                 pmax: float = None,
                 how: str = 'quartile'):
        self.num_list = num_list
        self.diff_num = diff_num
        self.pmin = pmin
        self.pmax = pmax
        self.how = how

    def fit(self, X, y=None):
        X = X.copy()
        cate_list, self.num_list = get_cate_num(X, None, self.num_list)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.num_list:
            describe_ = X[col].describe()
            fmin_, fmax_ = 0.0, 0.0
            if self.how == 'quartile':
                IQR = round(describe_['75%'] - describe_['25%'], 2)
                fmin_ = round(describe_['25%'] - 1.5 * IQR, 2)
                fmax_ = round(describe_['75%'] + 1.5 * IQR, 2)
            elif self.how == 'self_percent':
                fmin_ = round(X[col].quantile(self.pmin), 2)
                fmax_ = round(X[col].quantile(self.pmax), 2)
            elif self.how == 'mean':
                fmin_ = round(describe_['mean'] - 3 * describe_['std'], 2)
                fmax_ = round(describe_['mean'] + 3 * describe_['std'], 2)
            else:
                logging.error("don't have that method!")
            logging.info('deal with "' + col + '" lower fmin size: ' + str(X.loc[X[col] < fmin_, col].shape[0]))
            logging.info('deal with "' + col + '" higher fmax size: ' + str(X.loc[X[col] > fmax_, col].shape[0]))
            X.loc[X[col] < fmin_, col] = fmin_
            X.loc[X[col] > fmax_, col] = fmax_
        logging.info('fix_outlier Success!!!')
        return X


class binning(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 diff_num: int = 10,
                 how: str = 'dtc',
                 q=5,
                 include_y: bool = True,
                 random_state: int = None):
        self.num_list = num_list
        self.diff_num = diff_num
        self.how = how
        self.q = q  # just for equal_size and equal_width method
        self.include_y = include_y
        self.random_state = random_state
        self.cut_dict = None

    def fit(self, X, y=None):
        X = X.copy()
        cate_list, self.num_list = get_cate_num(X, None, self.num_list)
        if len(self.num_list) == 0:
            return self

        if self.include_y:
            X['y'] = y

        if self.how == 'dtc':
            check_data_y(X)
            self.cut_dict = {}
            for num in self.num_list:
                """
                    利用决策树获得最优分箱的边界值列表
                """
                boundary = []  # 待return的分箱边界值列表
                clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                             max_leaf_nodes=6,  # 最大叶子节点数
                                             min_samples_leaf=0.05,  # 叶子节点样本数量最小占比
                                             random_state=self.random_state)  # 随机种子
                clf.fit(X[num].values.reshape(-1, 1), X['y'])  # 训练决策树

                n_nodes = clf.tree_.node_count
                children_left = clf.tree_.children_left
                children_right = clf.tree_.children_right
                threshold = clf.tree_.threshold

                for i in range(n_nodes):
                    if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                        boundary.append(threshold[i])

                boundary.sort()

                min_x = np.around(X[num].min() - 1e-2, 3)
                max_x = np.around(X[num].max(), 3)
                boundary = [min_x] + boundary + [max_x]
                self.cut_dict[num] = boundary

        elif self.how == 'mono':
            self.cut_dict = {}
            check_data_y(X)
            from xverse.transformer import MonotonicBinning
            mono_bin = MonotonicBinning(feature_names=self.num_list)
            mono = mono_bin.fit(X.drop('y', axis=1), X['y'])
            self.cut_dict = mono.bins
            for v in self.cut_dict.values():
                v[0] -= 1e-2
        if self.include_y:
            X.drop('y', axis=1, inplace=True)
        return self

    def transform(self, X):
        if not self.num_list:
            return X
        X = X.copy()
        if self.how == 'dtc' or self.how == 'mono':
            for num in self.num_list:
                X[num] = pd.cut(x=X[num], bins=self.cut_dict[num])
                # labels=[chr(i) for i in range(97, 97 + len(self.cut_dict[num]) - 1)])  # 左开右闭
                X[num] = X[num].astype('object')

        elif self.how == 'equal_size':
            for num in self.num_list:
                X[num] = pd.qcut(x=X[num], q=self.q)
                X[num] = X[num].astype('object')

        elif self.how == 'equal_width':
            for num in self.num_list:
                X[num] = pd.cut(x=X[num], bins=self.q)
                X[num] = X[num].astype('object')
        else:
            logging.error("do not have that method, you can use 'dtc'/'mono'/'equal_size' or 'equal_width'")
            return X

        logging.info('binning Success!!!')
        return X


class dummies(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cate_list: list = None,
                 diff_num: int = 10,
                 dummy_na: bool = True,
                 ):
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.dummy_na = dummy_na
        self.columns2dummies = None

    def fit(self, X, y=None):
        import re
        X = X.copy()
        self.cate_list, num_list = get_cate_num(X, self.cate_list, None)
        X = pd.get_dummies(X, columns=self.cate_list, dummy_na=self.dummy_na, prefix=self.cate_list)
        regex = re.compile(r"\(|\]|\[\,", re.IGNORECASE)
        X.columns = [regex.sub("_", col) for col in X.columns.values]
        self.columns2dummies = X.columns
        return self

    def transform(self, X):
        import re
        X = X.copy()
        X = pd.get_dummies(X, columns=self.cate_list, dummy_na=self.dummy_na, prefix=self.cate_list)
        
        regex = re.compile(r"\(|\]|\[\,", re.IGNORECASE)
        X.columns = [regex.sub("_", col) for col in X.columns.values]

        fit_only = set(self.columns2dummies).difference(set(X.columns))
        trans_only = set(X.columns).difference(set(self.columns2dummies))

        if len(fit_only) != 0:
            for item in fit_only:
                X[item] = np.zeros((X.shape[0],), dtype=np.int)
        else:
            pass
        if len(trans_only) != 0:
            for item in trans_only:
                X.drop(item, axis=1, inplace=True)
        else:
            pass
        # if any(x in str(col) for x in {'(', '[', ']', ')', ','}) else col
        logging.info('dummies Success!!!')
        return X


class label_encoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 columns: list = None,
                 encode_na: bool = False
                 ):
        self.columns = columns
        self.encode_na = encode_na
        self.encoder_dict = {}

    def fit(self, X, y=None):
        X = X.copy()
        numerical_features = list(X._get_numeric_data().columns)
        self.columns = list(set(X.columns) - set(numerical_features))
        return self

    def transform(self, X):
        X = X.copy()
        from sklearn.preprocessing import LabelEncoder
        for col in self.columns:
            exist_na = False
            if X[col].isnull().any():  # 是否有空值
                X[col].fillna('NaN', inplace=True)
                exist_na = True
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            X[col] = X[col].astype('int64')
            self.encoder_dict[col] = dict(enumerate(le.classes_))
            if not self.encode_na and exist_na:
                X[col].replace({list(le.classes_).index('NaN'): np.nan}, inplace=True)
            else:
                pass
        logging.info('label_encoder Success!!!')
        return X


class scaler(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method: str = 'Standard',
                 num_list: list = None,
                 cate_list: list = None,
                 diff_num: int = 10
                 ):
        self.method = method
        self.num_list = num_list
        self.cate_list = cate_list
        self.diff_num = diff_num

    def fit(self, X, y=None):
        self.cate_list, self.num_list = get_cate_num(X, self.cate_list, self.num_list)
        return self

    def transform(self, X):

        X = X.copy()
        if self.method == 'Standard':
            from sklearn.preprocessing import StandardScaler
            X[self.num_list] = StandardScaler().fit_transform(X[self.num_list])
        elif self.method == 'MinMax':
            from sklearn.preprocessing import MinMaxScaler
            X[self.num_list] = MinMaxScaler().fit_transform(X[self.num_list])
        logging.info('scaler Success!!!')
        return X
