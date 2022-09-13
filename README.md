## Credit Card Transaction Fraud Detection


### Introduction
Fraud is one of the major problem for the banks and credit companies, both because of the large volume of transactions that are completed legitimately each day and also because many fraudulent transactions look a lot like normal transactions. While there is only a small fraction of transactions are fraudulent.

Identifying fraudulent transactions is a common type of imbalanced binary classification where the focus is on the negative class (fraud) class. The dependent variable is a sparce binomial distribution that majority of samples are not fraudulent.

In this project, I am going to conduct an example of a methodology for a data science project. This methodology considers several steps and assumptions:
    
   > 1. Entertaining a research problem, defining a research question, and providing a research plan
   > 2. Understanding the data and performing a data exploratory analysis to discover the relationships between variables
   > 3. Providing a reusable, modular, maintainable, and well-documented code
   > 4. Investigating different areas of machine learning (supervised, unsupervised, parametric, non-parametric, temporal/time-series, non-temporal) and relating the problem to each area when it is possible
   > 5. Evaluating and assessing the goodness of each model
   > 6. Providing conclusion, final thoughts, and possible future works for each area of work.
   > 7. In this project due to limited time, the goal is not providing the most accurate solutions, but investigating different possibilities. Therefore, any method such as cross validation, hyper parameter tuning, etc. which are involve to improve the robustness and/or performance of the models are not used in this scope of work. These methods are essentials and before attempting to integrate and deploy the model for real-time and actual use cases, need to be implemented by other data scientists.


### Installation
The project has been packaged with Python 3.7 and the dependencies are stored in the “requirements.txt” file. To install the project in a virtual environment, simply the below steps needs to be taken. [Installing Packages — Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)

   1. Install the latest version of Python 3.7.9:
    [Python Release Python 3.7.9 | Python.org](https://www.python.org/downloads/release/python-379/)
    
   2. Check python installation
    `python --version`
    
   3. Check “pip”
    `py -m pip --version`
    If it doesn’t work, please see the python packaging user guide
    
   4. Creating virtual environment
    `py -m venv <DIR>`
    
   5. Activating the virtual environment
   `<DIR>\Scripts\activate`
    
   6. Installing the project requirements from the “requirements.txt” in the root directory of project
    `py -m pip install -r requirements.txt`


### Modular Programming OOP/Modular

In this work, the focus was developing code using Python modular programming. In this way, we need to be ensured that simplicity, maintainablity, reusability, and scoping will be implemented during code development. ALso the code needs to be well-documented to pass to other members of team. The code is packaged inside "fraud_detection" directory. The strucutre of package and the modules are as follow:

```
 fraud_detection
 +-- __init__.py
 +-- main.py
 +-- eda
 |    |-- __init__.py
 |    |-- computations.py
 |    |-- eda.py
 |    |-- visualization.py
 +-- models
      |-- __init__.py
      |-- regression.py
      |-- supervised.py
      |-- unsupervised.py
      |-- time_series.py
 requirements.txt
```

### Notebooks
[1_EDA.ipynb](1_EDA.ipynb)

[2_Regression_Analysis.ipynb](2_Regression_Analysis.ipynb)

[3_Machine_Learning_Models.ipynb](3_Machine_Learning_Models.ipynb)
