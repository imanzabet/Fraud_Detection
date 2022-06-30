import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class computations(object):
    def __init__(self):
        pass
    
    def boxcox_prep(self, df):
        '''
        preperation of a dataframe column before boxcox transformation
        '''
        min_val = min(df)
        if min_val <= 0:
            return df - min_val + 0.01
        else:
            return df
    
    # Memory Optimization
    def reduce_mem_usage(self, df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and str(col_type) != 'category':
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:5] == 'float':
                    if len(set(df[col]) - set(np.array(df[col], int))) == 0:
                        df[col] = df[col].astype(np.int64)
                        col_type = df[col].dtype
                
                if str(col_type)[:3] == 'int':
                    if len(set(df[col])) < 100 :
                        df[col] = df[col].astype('category')
                        col_type = df[col].dtype
                        
                elif str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
    
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            elif str(col_type) != 'category':
                df[col] = df[col].astype('category')
    
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df
    
    def min_max_scalar(self, df, cols):
        sc = MinMaxScaler()

        df_sc = df.copy()
        df_sc[cols] = sc.fit_transform(df[cols])
        return df_sc
    
class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output
        

if (__name__ == '__main__'):
    cmp = computations()
    df = pd.DataFrame({'A': [1,2,3]})
    print(cmp.boxcox_prep(df['A']))