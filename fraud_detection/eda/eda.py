from .visualization import visualization
from .computations import computations

class eda(visualization, computations):
    def __init__(self, df):
        self.df = df


if (__name__ == '__main__'):
    eda = eda()