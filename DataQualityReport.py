import gc
import pandas as pd
import numpy as np
from pyecharts.charts import Bar, Page, Pie
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

pd.options.display.max_columns = None  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 取消科学计数法
CurrentConfig.ONLINE_HOST = "./pyecharts-assets-master/assets/"


def get_kind(series_, diff_limit):
    series_ = series_.astype('str')
    series_ = series_.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    series_.dropna(inplace=True)
    if series_.nunique() > diff_limit:
        kind = 'numeric'
    else:
        kind = 'categorical'
    return series_, kind


class DQReport(object):
    def __init__(self, data, target, Id=None, numeric_list: list = None, diff_limit=8, encoding='utf-8', k=5):
        """
            data:           input data-frame
            target:         data-frame's target
            diff_limit:     depend on number of different values, to distinguish numeric or categorical attribute
            k:              for numeric -- equal width binning -- amount of binning on html-showing
        """
        self.data = data.copy()
        self.target = target
        self.diff_limit = diff_limit
        self.categorical_list = []
        self.numeric_list = numeric_list
        self.k = k
        self.Id = Id
        self.target = target
        self.data_size = self.data.shape[0]
        self.encoding = encoding
        self.colname = list(self.data.columns)
        if self.Id is not None:
            self.colname.remove(self.Id)
        self.colname.remove(self.target)
        self.list_kind = []

        if self.numeric_list is None:
            self.numeric_list = self.data._get_numeric_data().columns
            self.numeric_list = [col for col in self.numeric_list if col != self.target]
            self.categorical_list = list(set(self.colname).difference(set(self.numeric_list)))
        else:
            self.categorical_list = list(set(self.colname).difference(set(self.numeric_list)))

        for col in self.categorical_list:
            self.data[col] = self.data[col].astype('object')

    def auto_NumericType(self, inplace=True):
        auto_numeric_list = []
        for col in list(self.colname):
            tmp_, kind = get_kind(self.data[col], self.diff_limit)
            if kind == 'numeric':
                auto_numeric_list.append(col)
            else:
                pass
        if inplace:
            self.numeric_list = auto_numeric_list
            self.categorical_list = list(set(self.colname).difference(set(self.numeric_list)))
        else:
            pass
        return auto_numeric_list

    def get_numeric_list(self):
        return self.numeric_list

    def get_categorical_list(self):
        return self.categorical_list

    def re_type(self):
        for col in self.colname:
            if col in self.categorical_list:
                self.data[col] = self.data[col].astype('object')
            else:
                self.data[col] = self.data[col].astype('float64')
        return self.data

    def create_graph(self, page_, data_, col, target, new_list_taget, pyecharts: bool = False, need_plt: bool = True,
                     col_type=None):
        import matplotlib.pyplot as plt
        """
        page_:      current interface
        col:        column name
        col_type:   numeric and categorical
        """
        file = data_.copy()
        file.sort_values(by=col, inplace=True)
        if col_type == 'numeric':
            file[col] = pd.cut(file[col], self.k, duplicates='drop')
        else:
            if file[col].nunique() > 15:
                file[col] = file[col].replace(
                    file[col].value_counts(dropna=False, ascending=False).index.tolist()[15:], 'Others(System)')

        file[col].replace(np.nan, 'unKnow', inplace=True)  # 數值屬性切割完成後需要把空值轉為unknown
        file[col] = file[col].astype('str')
        list_fea = list(file[col].unique())  # 類別欄位-list
        df_cut = file[[col, target]]
        df_cut.reset_index(inplace=True)

        col_ = df_cut.groupby([col, target], sort=False)['index'].count().unstack(fill_value=0).stack()
        # print('col_\n', col_)
        df_cut.drop('index', inplace=True, axis=1)
        col_sum = col_.groupby(level=0, sort=False).sum()
        # print('col_sum\n', col_sum)
        col_g = (col_ / col_sum).unstack()[new_list_taget]
        # print('col_g\n', col_g)

        col_value = col_g.values
        if need_plt:
            """
            圓餅圖
            pie:    Pyechart
            plt:    Matplotlib
            """
            plt.pie(x=np.around((col_sum / self.data_size * 100), 2), autopct='%.2f%%', labels=list_fea)
            plt.title(col)
            plt.show()

            """
            直條圖
            bar0:   Pyechart
            plt:    Matplotlib
            """
            ax = col_sum.reset_index().plot(kind='bar', rot=1, figsize=(8, 8), title=col)  # 百分比堆疊圖
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy()
                ax.text(x + width / 2,
                        y + height / 2,
                        '{:.1f}'.format(height),
                        horizontalalignment='center',
                        verticalalignment='center')
            ax.set_xticklabels(labels=list_fea, rotation=30)
            plt.show()

            """
            百分比堆疊圖
            bar1:   Pyechart
            plt:    Matplotlib
            """
            ax = col_g.plot(kind='bar', stacked=True, rot=1, figsize=(16, 8), title=col)  # 百分比堆疊圖
            for p in ax.patches:
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy()
                ax.text(x + width / 2,
                        y + height / 2,
                        '{:.1f} %'.format(height * 100),
                        horizontalalignment='center',
                        verticalalignment='center')
                ax.set_xticklabels(labels=list_fea, rotation=30)
            plt.show()
        if pyecharts:
            if col_type != 'numeric':
                pie = Pie(init_opts=opts.InitOpts(width='1000px', height='600px'))
                pie.set_global_opts(legend_opts=opts.LegendOpts(orient='vertical', pos_top='15%', pos_left='2%'))
                pie.add('', [list(z) for z in zip(list_fea, np.around((col_sum / self.data_size * 100), 2))],
                        is_clockwise=True)
                pie.set_global_opts(title_opts=opts.TitleOpts(title=col, subtitle='百分比對比圖(%)'))
                page_.add(pie)

            bar0 = Bar(init_opts=opts.InitOpts(width='1000px', height='600px'))  # 圖大小
            bar0.add_xaxis(list_fea)
            bar0.add_yaxis(col, list(col_sum))
            bar0.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            bar0.set_global_opts(title_opts=opts.TitleOpts(subtitle='柱狀統計圖'),
                                 datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100))
            page_.add(bar0)

            bar1 = Bar(init_opts=opts.InitOpts(width='1000px', height='600px'))  # 圖大小
            bar1.add_xaxis(list_fea)
            for tar_class, i in zip(new_list_taget, np.arange(0, len(new_list_taget))):
                list_class = list(np.around(col_value[:, i], 2))
                bar1.add_yaxis(tar_class, list_class, stack='stack1')

            bar1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            bar1.set_global_opts(title_opts=opts.TitleOpts(subtitle='百分比堆疊圖'),
                                 datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100))  # 数據窗口範圍的起始/终止百分比卷軸
            page_.add(bar1)

        # del file
        # gc.collect()
        return page_

    def SReport(self, save_path=None, top=5):
        """
            top:                get the n most frequent values,subset .value_counts() and grab the index
            save_path:          csv path
            list_na:            the count of NAN
            list_na_ratio:      the ratio of NAN
            list_value:         the count of non-null
            list_value_ratio    the ratio of non-null
            list_diff_count     the count of different value
            list_diff_value     show some different elements
        """
        list_na, list_na_ratio, list_value, list_value_ratio, list_diff_count, list_diff_value = [], [], [], [], [], []
        data = self.data.copy()
        list_col = [col for col in data.columns if col != self.Id]
        for col in list_col:
            diff_count = data[col].nunique()  # size of different number
            list_diff_count.append(diff_count)
            drop_na = data[col].dropna()
            dropna_size = drop_na.shape[0]  # size of after drop
            na_size = self.data_size - dropna_size  # size of nan
            list_na.append(na_size)
            na_ratio = str(round(na_size / self.data_size * 100, 4)) + '%'
            list_na_ratio.append(na_ratio)
            list_value.append(dropna_size)  # ratio of value
            value_ratio = str(round(dropna_size / self.data_size * 100, 4)) + '%'
            list_value_ratio.append(value_ratio)
            temp, kind = get_kind(series_=data[col], diff_limit=self.diff_limit)
            self.list_kind.append(kind)
            if kind == 'numeric':
                a = list(temp.unique())
                b = list(drop_na.unique())
                if len(a) == len(b):
                    if len(a) > 3:
                        list_diff_value.append(a[:3] + ['...'])
                    else:
                        list_diff_value.append(a)
                else:
                    list_diff_value.append(list(set(b).difference(set(a))) + a[:3] + ['...'])
            else:
                if diff_count <= top:
                    llist = drop_na.value_counts().index.tolist()
                else:
                    llist = drop_na.value_counts().index.tolist()[:top]
                    llist = llist + ['...']
                list_diff_value.append(llist)

        # release some RAM
        # del data
        # gc.collect()
        # make-up dataframe
        dict_data = {'col_name': list_col, 'kinds': self.list_kind, 'null': list_na, 'null_ratio': list_na_ratio,
                     'value': list_value, 'value_ratio': list_value_ratio, 'count of different kinds': list_diff_count,
                     'value of different': list_diff_value}
        data_quality_report_summary = pd.DataFrame(dict_data, columns=[
            'col_name', 'kinds', 'null', 'null_ratio', 'value', 'value_ratio', 'count of different kinds',
            'value of different'])
        if save_path is not None:
            data_quality_report_summary.to_csv(save_path, index=False)
        # print(data_quality_report_summary)
        return data_quality_report_summary

    def NReport(self, csv_save_path=None, pyecharts=False, need_plt=False, html_save_path='numeric_analyse.html'):
        data = self.data.copy()
        list_min, list_max, list_mean, list_std, list_mean_sub_std, list_mean_add_std, list_Q1, list_Q3, \
        list_quartile_min, list_quartile_max = [], [], [], [], [], [], [], [], [], []

        page = Page(layout=Page.SimplePageLayout)
        new_list_taget = data[self.target].value_counts(ascending=True).index  # 目标-list
        for col in list(self.numeric_list):
            page = self.create_graph(page_=page, data_=data, col=col, col_type='numeric', target=self.target,
                                     new_list_taget=new_list_taget, pyecharts=pyecharts, need_plt=need_plt)
            describe_ = data[col].describe()
            list_min.append(round(describe_['min'], 2))
            list_max.append(round(describe_['max'], 2))
            list_std.append(round(describe_['std'], 2))
            list_mean.append(round(describe_['mean'], 2))
            list_mean_sub_std.append(round(describe_['mean'] - 3 * describe_['std'], 2))
            list_mean_add_std.append(round(describe_['mean'] + 3 * describe_['std'], 2))
            list_Q1.append(round(describe_['25%'], 2))
            list_Q3.append(round(describe_['75%'], 2))
            IQR = round(describe_['75%'] - describe_['25%'], 2)
            list_quartile_min.append(round(describe_['25%'] - 1.5 * IQR, 2))
            list_quartile_max.append(round(describe_['75%'] + 1.5 * IQR, 2))
        if pyecharts:
            page.page_title = 'numeric'  # html标签
            page.render(path=html_save_path)

        dict_data = {'numeric_name': self.numeric_list, 'Min': list_min, 'Max': list_max, 'Mean': list_mean,
                     'StDev': list_std,
                     'M-3': list_mean_sub_std, 'M+3': list_mean_add_std, 'Q1': list_Q1, 'Q3': list_Q3,
                     'Q1-1.5*IQR': list_quartile_min, 'Q3+1.5*IQR': list_quartile_max}
        data_quality_report_numeric = pd.DataFrame(dict_data, columns=[
            'numeric_name', 'Min', 'Max', 'Mean', 'StDev', 'M-3', 'M+3', 'Q1', 'Q3', 'Q1-1.5*IQR', 'Q3+1.5*IQR'])

        if csv_save_path is not None:
            data_quality_report_numeric.to_csv(csv_save_path, index=False)
        # print(data_quality_report_numeric)

        del data
        gc.collect()
        return data_quality_report_numeric

    def CReport(self, csv_save_path=None, pyecharts=False, need_plt=False, html_save_path='categorical_analyse.html'):
        data = self.data.copy()
        page = Page(layout=Page.SimplePageLayout)  # the new page
        new_list_taget = data[self.target].value_counts(ascending=True).index  # 目標-list

        max_nunique = max([data[col].nunique() for col in self.categorical_list])  # 記錄csv表格最大寬度
        max_nunique = 16 if max_nunique >= 15 else max_nunique
        category_df = pd.DataFrame(columns=np.arange(max_nunique + 1))
        # category_df.to_csv(os.path.dirname(path + '/' +file_path) + '/ttttt.csv')
        for col in self.categorical_list:
            data[col] = data[col].astype('str')
            page = self.create_graph(page_=page, data_=data, col=col, col_type='categorical', target=self.target,
                                     pyecharts=pyecharts,
                                     new_list_taget=new_list_taget, need_plt=need_plt)
            if data[col].isnull().any():
                category_name = data[col].value_counts().index.tolist()
                category_name.append('NAN')
            else:
                category_name = data[col].value_counts().index.tolist()
            category_name = category_name[:15] if len(category_name) >= 15 else category_name
            if len(category_name) >= 15:
                percent_list = list(data[col].value_counts(normalize=True, dropna=False)[:15])
                percent_list = [str(round(i * 100, 5)) + '%' for i in percent_list]

                dict_data = {col: [col] + category_name + ['...'],
                             'counts': ['counts'] + list(data[col].value_counts(dropna=False))[:15] + ['...'],
                             'percent': ['percent'] + percent_list + ['...']
                             }
            else:
                percent_list = list(data[col].value_counts(normalize=True, dropna=False))
                percent_list = [str(round(i * 100, 5)) + '%' for i in percent_list]
                dict_data = {col: [col] + category_name,
                             'counts': ['counts'] + list(data[col].value_counts(dropna=False)),
                             'percent': ['percent'] + percent_list
                             }

            data_quality_report_categorical = pd.DataFrame(dict_data,
                                                           columns=[col, 'counts', 'percent']).values.T
            category_df = pd.concat(
                [category_df, pd.DataFrame(data_quality_report_categorical)])
        category_df.reset_index(drop=True, inplace=True)
        # print(category_df)

        if pyecharts:
            page.page_title = 'categorical'
            page.render(html_save_path)

        if csv_save_path is not None:
            category_df.to_csv(csv_save_path, encoding=self.encoding, index=False)
        return category_df

