import pandas as pd
import numpy as np
import os
import logging
import xgboost as xgb
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
#from keras.utils import plot_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

pd.options.display.max_columns = None  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # 取消科学计数法

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')
					
#os.environ["PATH"] += os.pathsep + 'C:/Users/leeys/Desktop/Python/Machine-learning-with-python-cookbook/Materials/14.Trees and Forests/Graphviz2.38/bin'


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


class feature_selection(BaseEstimator, TransformerMixin):
    from sklearn.tree import DecisionTreeClassifier

    def __init__(self,
                 num_list: list = None,
                 method: str = 'sys',
                 diff_num: int = 10,
                 pos_label: str = 1,
                 need_sort: bool = True,
                 sys_threshold: float = 0.1,
                 p_threshold: float = 0.05,
                 dtc_params: dict = None,
                 dtc_threshold: float = 0.1,
                 rf_params: dict = None,
                 rf_threshold: float = 0.1,
                 xgb_params: dict = None,
                 xgb_threshold: float = 0.1,
                 variance_threshold: float = 0,
                 corr_threshold: float = 0.95,
                 sfs_estimator=DecisionTreeClassifier(),
                 sfs_estimator_params=None,
                 sfs_n_features_to_select=0.8,
                 sfs_direction='forward',
                 sfs_cv: int = 5,
                 sfs_scoring: str = None,
                 rfe_estimator=DecisionTreeClassifier(),
                 rfe_estimator_params=None,
                 rfe_cv: int = 5,
                 rfe_step: int = 1,
                 rfe_scoring: str = 'neg_mean_squared_error',
                 show_df: bool = False):

        if sfs_estimator_params is None:
            sfs_estimator_params = {}
        else:
            sfs_estimator.set_params(**sfs_estimator_params)

        if rfe_estimator_params is None:
            rfe_estimator_params = {}
        else:
            rfe_estimator.set_params(**rfe_estimator_params)

        if dtc_params is None:
            dtc_params = {}
        else:
            pass

        if rf_params is None:
            rf_params = {}
        else:
            pass

        if xgb_params is None:
            xgb_params = {}
        else:
            pass

        self.num_list = num_list
        self.cate_list = None
        self.method = method
        self.diff_num = diff_num
        self.pos_label = pos_label
        self.need_sort = need_sort
        self.sys_threshold = sys_threshold
        self.p_threshold = p_threshold
        self.dtc_params = dtc_params
        self.dtc_threshold = dtc_threshold
        self.rf_params = rf_params
        self.rf_threshold = rf_threshold
        self.xgb_params = xgb_params
        self.xgb_threshold = xgb_threshold
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.sfs_estimator = sfs_estimator
        self.sfs_estimator_params = sfs_estimator_params
        self.sfs_n_features = sfs_n_features_to_select
        self.sfs_direction = sfs_direction
        self.sfs_cv = sfs_cv
        self.sfs_scoring = sfs_scoring
        self.rfe_estimator = rfe_estimator
        self.rfe_estimator_params = rfe_estimator_params
        self.rfe_cv = rfe_cv
        self.rfe_step = rfe_step
        self.rfe_scoring = rfe_scoring
        self.show_df = show_df
        self.select_list = []

    def fit(self, X, y=None):
        X = X.copy()
        from scipy import stats
        self.cate_list, self.num_list = get_cate_num(X, self.cate_list, self.num_list)
        X['y'] = y
        yes = X[X['y'] == self.pos_label]
        yes.reset_index(drop=True, inplace=True)
        no = X[X['y'] != self.pos_label]
        no.reset_index(drop=True, inplace=True)
        del X['y']
        sys_cate_list, anova_kf_obj, anova_kf_p_obj, t_kf_obj, t_kf_p_obj = [], [], [], [], []
        sys_num_list, t_list, p_value_list, anova_f_list, anova_p_list, dtc_num_list, rf_num_list, xgb_num_list, vt_num_list = [], [], [], [], [], [], [], [], []
        if self.method == 'sys' or self.show_df is True:
            sys_value_dict = {}
            for obj in self.cate_list:
                value_list = list(X[obj].unique())
                value_sum = 0
                for value in value_list:
                    support_yes = (yes[yes[obj] == value].shape[0] + 1) / (yes.shape[0] + 1)
                    support_no = (no[no[obj] == value].shape[0] + 1) / (no.shape[0] + 1)
                    confidence_yes = support_yes / (support_yes + support_no)
                    value_sum += abs(2 * confidence_yes - 1) * (X[X[obj] == value].shape[0] / X.shape[0])
                sys_cate_list.append(value_sum)
                if value_sum >= self.sys_threshold and self.method == 'sys':
                    sys_value_dict[obj] = value_sum
                    self.select_list.append(obj)

            for num in self.num_list:
                mean_c1 = no[num].mean()
                std_c1 = no[num].std()
                mean_c2 = yes[num].mean()
                std_c2 = yes[num].std()
                value_sum = abs(mean_c1 - mean_c2) / (std_c1 + std_c2) * 2
                sys_num_list.append(value_sum)
                if value_sum >= self.sys_threshold and self.method == 'sys':
                    sys_value_dict[num] = value_sum
                    self.select_list.append(num)

            if self.need_sort and self.method == 'sys':
                self.select_list = [k for k, v in
                                    sorted(sys_value_dict.items(), key=lambda item: item[1], reverse=True)]

        if self.method == 'anova_kf' or self.show_df is True:
            akf_value_dict = {}
            for obj in self.cate_list:
                df_obj = pd.get_dummies(X[obj], prefix=obj)
                df_obj['result'] = y
                df_obj = df_obj.groupby('result').sum()
                obs = df_obj.values
                kf = stats.chi2_contingency(obs)
                '''
                chi2: The test statistic
                p: p-value
                dof: Degrees of freedom
                expected: The expected frequencies, based on the marginal sums of the table.
                '''
                chi2, p, dof, expect = kf
                anova_kf_obj.append(chi2)
                anova_kf_p_obj.append(p)

                if p < self.p_threshold and self.method == 'anova_kf':
                    akf_value_dict[obj] = p
                    self.select_list.append(obj)

            for num in self.num_list:
                anova_f, p = stats.f_oneway(yes[num], no[num])
                anova_f_list.append(anova_f)
                anova_p_list.append(p)
                # print('attr=%s, anova_f=%.5f, anova_p=%.5f' % (num, anova_f, anova_p))
                if p < self.p_threshold and self.method == 'anova_kf':
                    akf_value_dict[num] = p
                    self.select_list.append(num)

            if self.need_sort and self.method == 'anova_kf':
                self.select_list = [k for k, v in
                                    sorted(akf_value_dict.items(), key=lambda item: item[1], reverse=True)]

        if self.method == 't_kf' or self.show_df is True:
            t_value_dict = {}
            for obj in self.cate_list:
                df_obj = pd.get_dummies(X[obj], prefix=obj)
                df_obj['result'] = y
                df_obj = df_obj.groupby('result').sum()
                obs = df_obj.values
                kf = stats.chi2_contingency(obs)
                '''
                chi2: The test statistic
                p: p-value
                dof: Degrees of freedom
                expected: The expected frequencies, based on the marginal sums of the table.
                '''
                chi2, p, dof, expect = kf
                t_kf_obj.append(chi2)
                t_kf_p_obj.append(p)

                if p < self.p_threshold and self.method == 't_kf':
                    t_value_dict[obj] = p
                    self.select_list.append(obj)
            for num in self.num_list:
                t_t, t_p = stats.ttest_ind(yes[num], no[num], equal_var=False, nan_policy='omit')  # 'omit'忽略nan值执行计算
                t_list.append(t_t)
                p_value_list.append(t_p)
                if t_p < self.p_threshold and self.method == 't_kf':
                    t_value_dict[num] = t_p
                    self.select_list.append(num)
            if self.need_sort and self.method == 't_kf':
                self.select_list = [k for k, v in
                                    sorted(t_value_dict.items(), key=lambda item: item[1], reverse=True)]

        if self.method == 'var' or self.show_df is True:
            from sklearn.feature_selection import VarianceThreshold
            vt = VarianceThreshold(threshold=self.variance_threshold)
            vt.fit(X[self.num_list])
            vt_num_features = X[self.num_list].columns[vt.get_support(indices=True)]
            vt_num_list = vt.variances_
			
            if self.method == 'var':
                self.select_list.extend(vt_num_features)
            else:
                pass
			
            if self.need_sort and self.method == 'var':
                self.select_list = np.array(self.select_list)[np.argsort(-np.array(vt_num_list))]
            else:
                pass

        if self.method == 'corr' or self.show_df is True:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            upper.fillna(0, inplace=True)
            # upper[col].any()是否全部小于threshold
            if self.method == 'corr':
                self.select_list = [col for col in upper.columns if all(upper[col] <= self.corr_threshold)]
            else:
                pass

        if self.method == 'dtc' or self.show_df is True:
            dtc_value_dict = {}
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier().set_params(**self.dtc_params)
            clf.fit(X, y)
            clf_importance = clf.feature_importances_
            
            #'''
            for idx, v in enumerate(clf_importance):
                dtc_num_list.append(v)
                
            dtc_threshold = np.mean(dtc_num_list)
            #'''
                    
            for idx, v in enumerate(clf_importance):
                #dtc_num_list.append(v)

                if self.method == 'dtc' and v >= self.dtc_threshold:
                    dtc_value_dict[X.columns[idx]] = v
                    self.select_list.append(X.columns[idx])
                else:
                    pass
            if self.need_sort and self.method == 'dtc':
                self.select_list = [k for k, v in
                                    sorted(dtc_value_dict.items(), key=lambda item: item[1], reverse=True)]
            

        if self.method == 'rf' or self.show_df is True:
            from sklearn.ensemble import RandomForestClassifier
            rf_value_dict = {}
            clf = RandomForestClassifier().set_params(**self.rf_params)
            clf.fit(X, y)
            clf_importance = clf
            
            '''
            for idx, v in enumerate(clf_importance):
                rf_num_list.append(v)
                
            rf_threshold = np.mean(rf_num_list)
            '''
            
            for idx, v in enumerate(clf_importance):
                rf_num_list.append(v)
                if self.method == 'rf' and v >= self.rf_threshold:
                    rf_value_dict[X.columns[idx]] = v
                    self.select_list.append(X.columns[idx])
                else:
                    pass
            if self.need_sort and self.method == 'rf':
                self.select_list = [k for k, v in
                                    sorted(rf_value_dict.items(), key=lambda item: item[1], reverse=True)]

        if self.method == 'xgb' or self.show_df is True:
            from xgboost import XGBClassifier
            xgb_value_dict = {}
            clf = XGBClassifier().set_params(**self.xgb_params)
            clf.fit(X, y)
            clf_importance = clf.feature_importances_
            for idx, v in enumerate(clf_importance):
                xgb_num_list.append(v)
                if self.method == 'xgb' and v >= self.xgb_threshold:
                    xgb_value_dict[X.columns[idx]] = v
                    self.select_list.append(X.columns[idx])
                else:
                    pass
            if self.need_sort and self.method == 'xgb':
                self.select_list = [k for k, v in
                                    sorted(xgb_value_dict.items(), key=lambda item: item[1], reverse=True)]

        if self.method == 'sfs':
            from sklearn.feature_selection import SequentialFeatureSelector as sfs
            import matplotlib.pyplot as plt
            sfs_clf = sfs(estimator=self.sfs_estimator, n_features_to_select=self.sfs_n_features,
                          scoring=self.sfs_scoring,
                          direction=self.sfs_direction, cv=self.sfs_cv)
            sfs_clf.fit(X, y)
            if self.method == 'sfs':
                self.select_list = X.columns[sfs_clf.get_support(indices=True)]

        if self.method == 'rfe':
            from sklearn.feature_selection import RFECV
            import matplotlib.pyplot as plt
            rfe_clf = RFECV(estimator=self.rfe_estimator, step=self.rfe_step, scoring=self.rfe_scoring,
                            cv=self.rfe_cv)
            rfe_clf.fit(X, y)
            rfe_rank = rfe_clf.ranking_
            if self.method == 'rfe':
                self.select_list = X.columns[rfe_clf.support_]
                print("REFCV Optimal number of features : %d" % rfe_clf.n_features_)

                plt.figure()
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (nb of correct classifications)")
                plt.plot(range(rfe_clf.min_features_to_select,
                               len(rfe_clf.grid_scores_) + rfe_clf.min_features_to_select),
                         rfe_clf.grid_scores_)
                plt.show()

        if self.show_df is True:
            dic1 = {'categorical': self.cate_list, 'sys_importance_': sys_cate_list, 'T-Kf-Value': t_kf_obj,
                    'T-Kf-P-Value': t_kf_p_obj, 'Anova-Kf-Value': anova_kf_obj, 'Anova-Kf-P-Value': anova_kf_p_obj}
            df = pd.DataFrame(dic1,
                              columns=['categorical', 'sys_importance_', 'T-Kf-Value', 'T-Kf-P-Value',
                                       'Anova-Kf-Value', 'Anova-Kf-P-Value'])
            df.sort_values(by='Anova-Kf-P-Value', inplace=True)
            print(df)
            dic2 = {'numeric': self.num_list, 'sys_importance_': sys_num_list, 'T-Value': t_list,
                    'P-value': p_value_list, 'Anova-F-Value': anova_f_list, 'Anova-P-value': anova_p_list,
                    'rf_importance_': rf_num_list, 'xgb_importance_': xgb_num_list, 'Vt_variance': vt_num_list}
            df = pd.DataFrame(dic2,
                              columns=['numeric', 'sys_importance_', 'T-Value', 'P-value', 'Anova-F-Value',
                                       'Anova-P-value', 'rf_importance_', 'xgb_importance_', 'Vt_variance'])
            df.sort_values(by='Anova-P-value', inplace=True)
            print(df)
        self.select_list = list(self.select_list)
        print('After select attr: ', self.select_list)
        print('Number of features selected: ', len(self.select_list))
        
        #print('決策樹重要屬性門檻:', dtc_threshold)
        
        return self

    def transform(self, X):
        X = X.copy()
        logging.info('feature select success!')
        return X[self.select_list]


class feature_transformation(BaseEstimator, TransformerMixin):

    def __init__(self,
                 method: str = 'default',
                 n_components = 0.9,
                 pca_whiten: bool = False,
                 kpca_Kernel: str = 'rbf',
                 kpca_gamma: float = None,
                 params_dict = None,
                 random_state: int = 0):
        if params_dict is None:
            params_dict = {}
        self.method = method
        self.n_components = n_components
        self.pca_whiten = pca_whiten
        self.kpca_Kernel = kpca_Kernel
        self.kpca_gamma = kpca_gamma

        self.params_dict = params_dict
        self.random_state = random_state
        self.feature_model = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.method == 'default' or self.method == 'pca':
            from sklearn.decomposition import PCA
            feature_pca = PCA(n_components=self.n_components, whiten=self.pca_whiten,
                              random_state=self.random_state).set_params(**self.params_dict).fit(X)
            self.feature_model = feature_pca
        elif self.method == 'k_pca':
            from sklearn.decomposition import KernelPCA
            feature_KPca = KernelPCA(kernel=self.kpca_Kernel, gamma=self.kpca_gamma,
                                     n_components=int(self.n_components), random_state=self.random_state).set_params(
                **self.params_dict).fit(X)
            self.feature_model = feature_KPca
        elif self.method == 'nmf':
            from sklearn.decomposition import NMF
            feature_nmf = NMF(n_components=int(self.n_components), random_state=self.random_state).set_params(
                **self.params_dict).fit(X)
            self.feature_model = feature_nmf

        elif self.method == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            if y is None:
                logging.error('Missing y variable')
            else:
                feature_lda = LinearDiscriminantAnalysis(n_components=int(self.n_components)).set_params(
                    **self.params_dict).fit(X=X, y=y)
                self.feature_model = feature_lda

        elif self.method == 't_sne':
            from sklearn.manifold import TSNE
            feature_tsne = TSNE(n_components=int(self.n_components), random_state=self.random_state).set_params(
                **self.params_dict).fit(X)
            self.feature_model = feature_tsne
        else:
            logging.error('Illegal method, Please use one of the following methods: default/pca/k_pca/lda/nmf/t_sne. ')
        return self

    def transform(self, X):
        X = X.copy()
        return self.feature_model.transform(X)

class feature_learning(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method: str = 'auto',
                 percentage_bottlenecks: float = 0.5,
                 activation: str = 'softmax',
                 loss: str = 'binary_crossentropy',  # categorical_crossentropy
                 optimizer: str = 'adam',
                 rbm_learning_rate: float = 0.01,
                 rbm_n_iter: int = 10,
                 rbm_n_components: float = 0.5,
                 nn_hidden_size: float = 0.5,
                 random_state: int = 0):
        self.method = method
        self.percentage_bottlenecks = percentage_bottlenecks
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.rbm_learning_rate = rbm_learning_rate
        self.rbm_n_iter = rbm_n_iter
        self.rbm_n_components = rbm_n_components
        self.nn_hidden_size = nn_hidden_size
        self.random_state = random_state
        self.model_ = None
        self.encoder_ = None

    def fit(self, X, y=None):
        X = X.copy()
        from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
        import keras
        n_inputs = X.shape[1]
        if self.method == 'auto':
            visible = Input(shape=(n_inputs,))
            # encoder level 1
            e = Dense(n_inputs * 2)(visible)
            e = BatchNormalization()(e)
            e = LeakyReLU()(e)
            # encoder level 2
            e = Dense(n_inputs)(e)
            e = BatchNormalization()(e)
            e = LeakyReLU()(e)
            # bottleneck
            n_bottleneck = int(n_inputs * self.percentage_bottlenecks)
            bottleneck = Dense(n_bottleneck)(e)
            # define decoder, level 1
            d = Dense(n_inputs)(bottleneck)
            d = BatchNormalization()(d)
            d = LeakyReLU()(d)
            # decoder level 2
            d = Dense(n_inputs * 2)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU()(d)
            # output layer
            output = Dense(n_inputs, activation=self.activation)(d)
            # define auto-encoder model
            model = keras.Model(visible, output)
            model.compile(optimizer=self.optimizer, loss=self.loss)
            self.model_ = model
            self.encoder_ = keras.Model(inputs=visible, output=bottleneck)
        elif self.method == 'rbm':
            from sklearn.neural_network import BernoulliRBM
            rbm_clf = BernoulliRBM(learning_rate=self.rbm_learning_rate, n_iter=self.rbm_n_iter,
                                   n_components=int(n_inputs * self.rbm_n_components), random_state=self.random_state, verbose=True)
            rbm_clf.fit(X, y)
            self.model_ = rbm_clf
            self.encoder_ = rbm_clf
        elif self.method == 'nn':
            y_dummy = pd.get_dummies(y)
            visible = Input(shape=(n_inputs,))
            # hidden level 1
            h = Dense(int(n_inputs * self.nn_hidden_size))(visible)
            h = BatchNormalization()(h)
            h = LeakyReLU()(h)
            # output layer
            output = Dense(len(set(y_dummy)), activation=self.activation)(h)
            # define auto-encoder model
            model = keras.Model(visible, output)
            model.compile(optimizer=self.optimizer, loss=self.loss)
            self.model_ = model
            self.encoder_ = keras.Model(inputs=visible, output=h)

    def transform(self, X):
        X = X.copy()
        if self.method == 'auto' or self.method == 'nn':
            logging.info('feature_learning autoencoder Success!!!')
            return self.encoder_.predict(X)
        elif self.method == 'rbm':
            logging.info('feature_learning rbm Success!!!')
            return self.encoder_.transform(X)
        else:
            logging.error('Illegal method, Please use one of the following methods: auto/nn/rbm. ')

