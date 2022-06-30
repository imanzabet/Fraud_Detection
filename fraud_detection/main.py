import numpy as np
from scipy.stats import norm
import pandas as pd
import pickle


class preprocessing(object):
    def __init__(self):
        self.df = None
        self.path = 'C:/Users/Iman/Downloads/'

    def loading_dataset(self, fname, path=None):
        '''
        loading the JSON format of dataset into a Pandas Dataframe
        '''
        if path == None:
            path = self.path
        self.df = pd.read_json(path + fname, lines=True)
        return self.df

    def pickle_save(self, df, fname, path=None):
        '''
        Saving pickle format of dataset
        '''
        if path == None:
            path = self.path
        dfFile = open(path + fname, 'wb')
        pickle.dump(df, dfFile)
        dfFile.close()

    def pickle_load(self, fname, path=None):
        '''
        Loading pickle format of dataset
        '''
        if path == None:
            path = self.path
        dfFile = open(path + fname, "rb")
        return pickle.load(dfFile)


if (__name__ == '__main__'):
    df = preprocessing().loading_dataset(fname='transactions_s.txt')
    print(df)

    pp = preprocessing()
    pp.pickle_save(df, fname='dfPickle', path='./')
