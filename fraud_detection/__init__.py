print(f'Invoking __init__.py for {__name__}')
#import pkg.mod1, pkg.mod2

from .main import preprocessing




if (__name__ == '__main__'):
    print('Executing as standalone script')


'''
from os import chdir, getcwd
getcwd()
chdir('d:/projects/fraud detection/fraud_detection')
'''