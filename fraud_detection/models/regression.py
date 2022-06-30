import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import FTestAnovaPower

def descriptive_stats(data):
    #define given sample data
    #Calculate the sample parameters
    confidenceLevel = 0.95             #95% CI given
    degrees_freedom = len(data)-1 # len(set(data))-1      #degree of freedom = unique sample size-1
    sampleMean = np.mean(data)          #sample mean
    sampleStandardError = st.sem(data)   #sample standard error
    #create 95% confidence interval for the population mean
    confidenceInterval = st.t.interval(alpha=confidenceLevel, df=degrees_freedom, loc=sampleMean, scale=sampleStandardError)
    #print the 95% confidence interval for the population mean

    print("Sample size:", len(data))
    print("Degree of Freedom", degrees_freedom)
    print("Mean of Sample:", sampleMean)
    print("Standard Error of Sample:", sampleStandardError)
    print('The 95% confidence interval for the sample mean:',confidenceInterval)

# Detecting and Removing Multicollinearity
def calculate_vif(x):
    thresh = 5.0
    
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:", str(a), x.columns[a], str(max(vif)))
        print()
        if vif[a] <= thresh :
            break
        if i == 1 :          
            x = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
        elif i > 1 :
            x = x.drop(x.columns[a],axis = 1)
            vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    return(x)


# multicollinearity 3
def calculate_vif2(df):
      
    # the independent variables set
    # df
      
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
      
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                              for i in range(len(df.columns))]
      
    return vif_data.sort_values(by=['VIF'])

## Plotting multiple plots same figure
def residual_plot(model):
    fig, (axL, axR) = plt.subplots(2, figsize=(15, 15))
    plt.suptitle("Logistic Regression Residual Plots")


    # Deviance Residuals
    sns.regplot(model.fittedvalues, model.resid_dev, ax= axL,
                color="black", scatter_kws={"s": 5},
                line_kws={"color":"b", "alpha":1, "lw":2}, lowess=True)

    axL.set_title("Deviance Residuals \n against Fitted Values")
    axL.set_xlabel("Linear Predictor Values")
    axL.set_ylabel("Deviance Residuals")

    # Studentized Pearson Residuals
    sns.regplot(model.fittedvalues, model.resid_pearson, ax= axR,
                color="black", scatter_kws={"s": 5},
                line_kws={"color":"g", "alpha":1, "lw":2}, lowess=True)

    axR.set_title("Studentized Pearson Residuals \n against Fitted Values")
    axR.set_xlabel("Linear Predictor Values")
    axR.set_ylabel("Studentized Pearson Residuals")

    plt.show()

def conf_mat_plot(cnf_matrix, class_names=[0,1]):
    
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def roc_plot(log_reg, x_test, y_test):
    fpr, tpr, _ = metrics.roc_curve(y_test,  log_reg.predict(x_test))
    auc = metrics.roc_auc_score(y_test, log_reg.predict(x_test))
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
# ANOVA Power Test
def anova_power_test():
    # parameters for power analysis
    effect_sizes = array([0.08, 0.2, 0.5, 0.8])
    sample_sizes = array(range(5, 50))
    
    # calculate power curves from multiple power analyses
    analysis = FTestAnovaPower()
    analysis.plot_power(dep_var='nobs', 
                        nobs=sample_sizes, 
                        effect_size=effect_sizes,
                        alpha=0.05,
                        title='Power of Independent Samples F-Test ANOVA \n$\\alpha = 0.05$')
    pyplot.show()

if (__name__ == '__main__'):
    pass