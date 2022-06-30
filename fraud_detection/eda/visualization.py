from scipy.stats import boxcox
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

class visualization(object):
    def __init__(self):
        pass
    def plotScatterMatrix(self, df, plotSize, textSize):
        '''
        Scatter and density plots
        '''
        df = df.select_dtypes(include =[np.number]) # keep only numerical columns
        # Remove rows and columns that would lead to df being singular
        df = df.dropna('columns')
        df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
        columnNames = list(df)
        if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
            columnNames = columnNames[:10]
        df = df[columnNames]
        ax = pd.plotting.scatter_matrix(df, alpha=0.75, 
                                        figsize=[plotSize, plotSize], diagonal='kde')
        corrs = df.corr().values
        for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
            ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], 
                              (0.8, 0.2), xycoords='axes fraction', 
                              ha='center', va='center', size=textSize)
        plt.suptitle('Scatter and Density Plot')
        plt.show()
        
    def res_plot(self, col):
        fig = plt.figure(figsize=(5,8))
    
        col_log = np.log(col)
        col_boxcox = boxcox(col)[0]
    
        ax1 = fig.add_subplot(311)
        prob = stats.probplot(x=col, dist=stats.norm, plot=ax1)
        ax1.set_xlabel('')
        ax1.set_title('Normality of "No transform"')
    
        ax2 = fig.add_subplot(312)
        prob = stats.probplot(x=col_log, dist=stats.norm, plot=ax2)
        ax2.set_xlabel('')
        ax2.set_title('Normality of "Log transform"')
    
        ax3 = fig.add_subplot(313)
        prob = stats.probplot(x=col_boxcox, dist=stats.norm, plot=ax3)
        ax3.set_xlabel('')
        ax3.set_title('Normality of "BoxCox transform"')
    
    
        plt.show()
    
    def multi_dist_plot(self, df, nrows=2, ncols=2, figsize=(15,10)):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        df = df.dropna()
        col_names = df.columns.tolist()
        cnt = 0
        
        for i in range(nrows):
            for j in range(ncols):
                sns.distplot(df.iloc[:, cnt], ax=ax[i][j])
                ax[i][j].set_xlabel(col_names[cnt], fontsize=12)
                cnt+=1
                print(cnt)

        fig.suptitle('Data Distribution', fontsize=20)
        fig.subplots_adjust(top=0.95)
    
    def all_kde(self, df):
        var = df.columns.values

        i = 0
        t0 = df.loc[df['Class'] == 0]
        t1 = df.loc[df['Class'] == 1]
        
        sns.set_style('whitegrid')
        plt.figure()
        fig, ax = plt.subplots(8,4,figsize=(16,28))
        
        for feature in var:
            i += 1
            plt.subplot(8,4,i)
            sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
            sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            locs, labels = plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show();
    def all_boxplot(self, df, col_class, nrows=2, ncols=3):
        '''
        plotting the boxplot for aall the dataframe (df) columns
        '''
        col_names = list(df.columns)
        col_class = 'isFraud'
        nrows = 2
        ncols = 3
        #visualizing the features w high negative correlation
        f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25,15))
        
        f.suptitle('Boxplot Diagram', size=35)
        for i, feature in enumerate(col_names):
            row = i//ncols
            col = i%ncols
        #     print('{}, {}'.format(row, col))
            sns.boxplot(x=col_class, y=feature, data=df, ax=axes[row,col])
            
    def multiple_plotly_boxplot(self, df, rows=1):
        '''
        '''
        vars = list(df.columns)
        fig = make_subplots(rows=rows, cols=len(vars))
        for i, var in enumerate(vars):
            fig.add_trace(
                go.Box(y=df[var],
                    name=var,
                    hoverinfo='none'
                ),
                
                row=1, col=i+1
            )
    
        fig.update_traces(boxpoints='all', jitter=.3)
        fig.show()
    def hist_plot(self, df, figsize=(8,5)):
        '''
        plotting single histogram
        '''
        kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    
        plt.figure(figsize=figsize, dpi= 80)
        sns.distplot(df, color="dodgerblue", label="Compact", **kwargs)
        # sns.distplot(x2, color="orange", label="SUV", **kwargs)
        # sns.distplot(x3, color="deeppink", label="minivan", **kwargs)
        # plt.xlim(50,75)
        plt.legend()
        plt.show()

    def count_plot(self, df, col, hue=None,  figsize=(8,5)):
        '''
        plotting count plot of a dataframe
        :param df:
        :param figsize:
        :return:
        '''
        plt.figure(figsize=figsize, dpi=80)
        sns.countplot(x=col, hue=hue, data=df)
    def kde_plot(self, df, x, hue):
        sns.kdeplot(data=df, x=x, hue=hue)
        plt.show()
    
    def barplot_percnt(self, df, x, y, hue, label=True, 
                       figsize=(10,8), transposed=False):
        '''
        normalized values of categories
        '''
        sns.set(rc={'figure.figsize':figsize})
        perc_df = (df[x]
                   .groupby(df[hue])
                   .value_counts(normalize=True)
                   .rename('Percent')
                   .reset_index())
        
        if transposed:
            ax = sns.barplot(x='Percent', y=x, hue=hue, data=perc_df)
        else:
            ax = sns.barplot(x=x, y='Percent', hue=hue, data=perc_df)
        if label:
            for container in ax.containers:
                ax.bar_label(container)

    def feature_importance(self, pipeline: Pipeline, X_train: pd.DataFrame):
        """ Plots feature importance. Assumes a Pipeline, with classifier as name"""
        importance = pd.Series(pipeline.best_estimator_.named_steps["classifier"].feature_importances_,
                               index=X_train.columns)
        plt.figure(figsize=(10, 5))
        importance.sort_values(ascending=False).plot.bar()
        plt.show()