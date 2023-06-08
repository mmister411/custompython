##Standard Mr. Mister Imports 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

### Modeling & Machine Learning
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

### SKLearn Processing 
from sklearn import preprocessing as prep
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

####Other Modeling 
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, skew
from scipy import stats

###TRAIN-EVAL 
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split, ParameterGrid
from sklearn.metrics import auc, roc_curve, make_scorer, accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn import model_selection

##SpatioTemporal 
# import shapefile  #ESRI ShapeFile library 
# from pykml import parser as kmlparser 
# from pykml.factory import KML_ElementMaker as KML, ATOM_ElementMaker as ATOM, GX_ElementMaker as GX 
# from fbprophet import Prophet 

### HTML, XML, & SQL  
# from IPython.core.display import HTML  #HTML Rendering 
# from bs4 import BeautifulSoup  #Web scraping 
# import sqlalchemy  # For SQLengine 
# import pysql # for MySQL 
# import xml.etree.ElementTree as et  

#Seaborn
color = sns.color_palette()
sns.set_style('darkgrid')
#MatPlotLib

#Warning Errors
import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

#Pandas
pd.set_option('display.float_format', lambda x: '{:.7f}'.format(x)) #Limiting floats output to 4 decimal points


### NA/Null Handling
neginf = (0 - np.inf)


def check_na(df):
    """
    Check the percentage of NaN/-Inf values in each column of the DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A DataFrame containing the total count and percentage of NaN/-Inf values for each column, sorted in descending order.

    Example:
        >>> check_na(df)
                    NaNTotal  NanPercent
        column_name
        column1           50      0.5000
        column2           30      0.3000
        column3           10      0.1000
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['NaNTotal', 'NanPercent'])
    return missing_data


def mass_na_transform(df, cat_na_value='None_Na', mean=True, alibi=[]):
    """
    Transform NaN values in a DataFrame by replacing them with a specified categorical filler value or numerical mean/median.

    Parameters:
        df (DataFrame): The input DataFrame.
        cat_na_value (str): The categorical filler value to use for replacing NaN values in categorical columns. Defaults to 'None_Na'.
        mean (bool): A flag indicating whether to replace NaN values in numerical columns with the mean (True) or median (False) of each column. Defaults to True.
        alibi (list): A list of column names to be excluded from the transformation.

    Returns:
        DataFrame: The transformed DataFrame with NaN values replaced.

    Example:
        >>> transformed_df = mass_na_transform(df, cat_na_value='Unknown', mean=True, alibi=['column1'])
    """
    isnull_columns = pd.Series(df.isna().any())[pd.Series(df.isna().any()) == True].index.tolist()
    isNOTnull_columns = pd.Series(df.notna().any())[pd.Series(df.notna().any()) == True].index.tolist()
    nacheck = [isnull_columns, isNOTnull_columns]
    cat_na_value = cat_na_value
    na_categorical = [val for val in nacheck[0] if val not in df[nacheck[0]].describe().columns.tolist()]
    na_numerical = [val for val in nacheck[0] if val in df[nacheck[0]].describe().columns.tolist()]
    if mean == True:
        for i in na_numerical:
            if i not in alibi:
                df[i] = df[i].replace(neginf, np.nan).fillna(df[i].mean())
    else:
        for i in na_numerical:
            if i not in alibi:
                df[i] = df[i].replace(neginf, np.nan).fillna(df[i].quantile(.5))
    for i in na_categorical:
        if i not in alibi:
            df[i] = df[i].fillna(cat_na_value)

    return df


### Column Lists
def export_columns_lists(df):
    """
    Export lists of numerical and categorical column names from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary with 'num' as the key for the list of numerical column names and 'cat' as the key for the list of categorical column names.

    Example:
        >>> column_lists = export_columns_lists(df)
        >>> print(column_lists['num'])
        ['column1', 'column2', 'column3']
        >>> print(column_lists['cat'])
        ['column4', 'column5', 'column6']
    """
    numerical_column_lists = df.describe().columns.tolist()
    categorical_columns_list = df.drop(df.describe().columns.tolist(), axis=1).columns.tolist()
    return {'num': numerical_column_lists, 'cat': categorical_columns_list}


### Metrics
def corr_table(df, y_name):
    """
    Compute the correlation table of a DataFrame with a specified target column.

    Parameters:
        df (DataFrame): The input DataFrame.
        y_name (str): The name of the target column.

    Returns:
        DataFrame: The correlation table sorted in descending order based on the correlation with the target column.

    Example:
        >>> correlation_table = corr_table(df, 'target')
    """
    return df.corr().sort_values(by=y_name, axis=0, ascending=False).sort_values(by=y_name, axis=1, ascending=False)


def correlation_list(df, yvar, topnum=20):
    """
    Get a list of columns with the highest correlation to a specified target column.

    Parameters:
        df (DataFrame): The input DataFrame.
        yvar (str): The name of the target column.
        topnum (int): The number of top correlated columns to return. Defaults to 20.

    Returns:
        list: A list of column names with the highest correlation to the target column.

    Example:
        >>> correlated_columns = correlation_list(df, 'target', topnum=10)
    """
    return df.corr()[yvar].sort_values(ascending=False).head(topnum).index.tolist()[1:]


def get_t_stats(df, xlist, yvalue):
    """
    Compute t-statistics for the relationship between a list of independent variables and a dependent variable.

    Parameters:
        df (DataFrame): The input DataFrame.
        xlist (list): The list of independent variable column names.
        yvalue (str): The name of the dependent variable column.

    Returns:
        DataFrame: A DataFrame containing the t-statistics, standard errors, p-values, and t-values for each independent variable.

    Example:
        >>> t_stats = get_t_stats(df, ['column1', 'column2', 'column3'], 'target')
    """
    y = df[yvalue]
    listnum = 0
    statslist = []
    for i in xlist:
        X2 = sm.add_constant(df[i])
        est2 = sm.OLS(y, X2).fit()
        statslist.append(pd.DataFrame([est2.summary().tables[1][2].data], columns=est2.summary().tables[1][0].data))
        statslist[listnum]['Lable'] = statslist[listnum]['']
        statslist[listnum] = statslist[listnum].drop('', axis=1).transpose()[0].rename(
            statslist[listnum].transpose()[0].loc[['Lable']].values[0])
        listnum += 1

    df2 = pd.DataFrame(statslist)
    df2['PValue'] = df2['P>|t|']
    df2['StdERR'] = df2['std err']
    df2['TValue'] = df2['t']
    df3 = df2.drop(['Lable', 'std err', 't', '[0.025', 'P>|t|', '0.975]'], axis=1).sort_values(by='PValue').sort_values(by='PValue', ascending=False)
    return df3

##Standard Mr. Mister Imports 
import numpy as np
import pandas as pd
# import os, sympy as sym 
# from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB

### Modeling & Machine Learning
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

### SKLearn Processing 
from sklearn import preprocessing as prep
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

####Other Modeling 
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, skew
from scipy import stats
#import networkx as nx  # Graph Theory  
####Reporting and Visualization 

##Visualization
import seaborn as sns
import matplotlib.pyplot as plt
 
###TRAIN-EVAL 
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split, ParameterGrid
from sklearn.metrics import auc, roc_curve, make_scorer, accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn import model_selection


##SpatioTemporal 
#import shapefile  #ESRI ShapeFile library 
#from pykml import parser as kmlparser 
#from pykml.factory import KML_ElementMaker as KML, ATOM_ElementMaker as ATOM, GX_ElementMaker as GX 
#from fbprophet import Prophet 

### HTML, XML, & SQL  
#from IPython.core.display import HTML  #HTML Rendering 
#from bs4 import BeautifulSoup  #Web scraping 
#import sqlalchemy  # For SQLengine 
#import pysql # for MySQL 
#import xml.etree.ElementTree as et  

#Seaborn
color = sns.color_palette()
sns.set_style('darkgrid')
#MatPlotLib

#Warning Errors
import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

#Pandas
pd.set_option('display.float_format', lambda x: '{:.7f}'.format(x)) #Limiting floats output to 4 decimal points

### NA/Null Handling
neginf = (0-np.inf)

def check_na(df):
    """
    Compute the percentage of NaN/-Inf values in each column of a Pandas DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A DataFrame containing the total count and percentage of NaN/-Inf values in each column, sorted in descending order.

    Example:
        >>> missing_data = check_na(df)
        >>> print(missing_data)
                   NaNTotal  NanPercent
        column1          100     0.100000
        column2           50     0.050000
        column3            0     0.000000
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum() / df.shape[0])).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['NaNTotal', 'NanPercent'])
    return missing_data

def mass_na_transform(df, cat_na_value='None_Na', mean=True, alibi=[]):
    """
    Fill missing values in a Pandas DataFrame with either the mean or median for numerical columns, and a categorical filler value for categorical columns.

    Parameters:
        df (DataFrame): The input DataFrame.
        cat_na_value (str): The categorical filler value. Defaults to 'None_Na'.
        mean (bool): If True, fill missing values in numerical columns with the mean; if False, fill with the median. Defaults to True.
        alibi (list): List of column names to be excluded from the transformation. Defaults to an empty list.

    Returns:
        DataFrame: The transformed DataFrame with missing values filled.

    Example:
        >>> df_filled = mass_na_transform(df, cat_na_value='Unknown', mean=True, alibi=['column1', 'column2'])
    """
    isnull_columns = pd.Series(df.isna().any())[pd.Series(df.isna().any()) == True].index.tolist()
    isNOTnull_columns = pd.Series(df.notna().any())[pd.Series(df.notna().any()) == True].index.tolist()
    nacheck = [isnull_columns, isNOTnull_columns]
    cat_na_value = cat_na_value
    na_categorical = [val for val in nacheck[0] if val not in df[nacheck[0]].describe().columns.tolist()]
    na_numerical = [val for val in nacheck[0] if val in df[nacheck[0]].describe().columns.tolist()]
    if mean:
        for i in na_numerical:
            if i not in alibi:
                df[i] = df[i].replace(neginf, np.nan).fillna(df[i].mean())
    else:
        for i in na_numerical:
            if i not in alibi:
                df[i] = df[i].replace(neginf, np.nan).fillna(df[i].quantile(.5))
    for i in na_categorical:
        if i not in alibi:
            df[i] = df[i].fillna(cat_na_value)

    return df

### Column Lists
def export_columns_lists(df):
    """
    Export numerical and categorical column lists from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary with keys 'num' and 'cat', containing lists of numerical and categorical column names, respectively.

    Example:
        >>> column_lists = export_columns_lists(df)
        >>> numerical_columns = column_lists['num']
        >>> categorical_columns = column_lists['cat']
    """
    numerical_column_lists = df.describe().columns.tolist()
    categorical_columns_list = df.drop(df.describe().columns.tolist(), axis=1).columns.tolist()
    return {'num': numerical_column_lists, 'cat': categorical_columns_list}

### Metrics
def corr_table(df, y_name):
    """
    Compute the correlation table of a DataFrame with respect to a target column.

    Parameters:
        df (DataFrame): The input DataFrame.
        y_name (str): The name of the target column.

    Returns:
        DataFrame: A correlation table sorted in descending order of correlation values with respect to the target column.

    Example:
        >>> correlation_table = corr_table(df, 'target')
    """
    return df.corr().sort_values(by=y_name, axis=0, ascending=False).sort_values(by=y_name, axis=1, ascending=False)

def correlation_list(df, yvar, topnum=20):
    """
    Compute the top correlated columns with respect to a target column.

    Parameters:
        df (DataFrame): The input DataFrame.
        yvar (str): The name of the target column.
        topnum (int): The number of top correlated columns to return. Defaults to 20.

    Returns:
        list: A list of top correlated column names.

    Example:
        >>> correlated_columns = correlation_list(df, 'target', topnum=10)
        >>> print(correlated_columns)
        ['column1', 'column2', 'column3', ...]
    """
    return df.corr()[yvar].sort_values(ascending=False).head(topnum).index.tolist()[1:]

def get_t_stats(df, xlist, yvalue):
    """
    Compute the t-statistics for a list of predictor variables with respect to a target variable.

    Parameters:
        df (DataFrame): The input DataFrame.
        xlist (list): A list of predictor variable names.
        yvalue (str): The name of the target variable.

    Returns:
        DataFrame: A DataFrame containing the t-statistics for each predictor variable.

    Example:
        >>> t_stats = get_t_stats(df, ['predictor1', 'predictor2', 'predictor3'], 'target')
    """
    y = df[yvalue]
    listnum = 0
    statslist = []
    for i in xlist:
        X2 = sm.add_constant(df[i])
        est2 = sm.OLS(y, X2).fit()
        statslist.append(pd.DataFrame([est2.summary().tables[1][2].data], columns=est2.summary().tables[1][0].data))
        statslist[listnum]['Lable'] = statslist[listnum]['']
        statslist[listnum] = statslist[listnum].drop('', axis=1).transpose()[0].rename(statslist[listnum].transpose()[0].loc[['Lable']].values[0])
        listnum += 1
        
    df2 = pd.DataFrame(statslist)
    df2['PValue'] = df2['P>|t|']
    df2['StdERR'] = df2['std err']
    df2['TValue'] = df2['t']
    df3 = df2.drop(['Lable', 'std err', 't', '[0.025', 'P>|t|', '0.975]'], axis=1).sort_values(by='PValue').sort_values(by='PValue', ascending=False)
    return df3

def describe2(df, todrop=False):
    """
    Customized version of the describe() function in pandas.DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        todrop (bool): If True, drop specified columns from the description. Defaults to False.

    Returns:
        DataFrame: A DataFrame containing the descriptive statistics, kurtosis, skewness, variance, and correlation with the target column.

    Example:
        >>> description = describe2(df, todrop=False)
    """
    if todrop == False:
        return df.describe().append(pd.DataFrame([df.kurt().rename("kurtosis"), df.skew().rename("skew"), df.var().rename('Variance'), df.corr()[y_name].rename('Y_Correlation')])).transpose().sort_values(by='Y_Correlation', ascending=False)
    else:
        return df.describe().append(pd.DataFrame([df.kurt().rename("kurtosis"), df.skew().rename("skew"), df.var().rename('Variance'), df.corr()[y_name].rename('Y_Correlation')])).drop(todrop, axis=1).transpose().sort_values(by='Y_Correlation', ascending=False)

def subgroup_median_fillna(df, target, grouper, newcolumn=False):
    """
    Fill missing values in a column by the median value within each subgroup defined by another column.

    Parameters:
        df (DataFrame): The input DataFrame.
        target (str): The name of the column to fill missing values.
        grouper (str): The name of the column defining subgroups.
        newcolumn (bool or str): If False, fill missing values in the original column; if a string, create a new column to store the filled values. Defaults to False.

    Example:
        >>> subgroup_median_fillna(df, 'target', 'group')
    """
    if newcolumn == False:
        df[target] = df.groupby(grouper)[target].transform(lambda x: x.fillna(x.median()))
    else:
        df[newcolumn] = df.groupby(grouper)[target].transform(lambda x: x.fillna(x.median()))

def subgroup_mean_fillna(df, target, grouper, newcolumn=False):
    """
    Fill missing values in a column by the mean value within each subgroup defined by another column.

    Parameters:
        df (DataFrame): The input DataFrame.
        target (str): The name of the column to fill missing values.
        grouper (str): The name of the column defining subgroups.
        newcolumn (bool or str): If False, fill missing values in the original column; if a string, create a new column to store the filled values. Defaults to False.

    Example:
        >>> subgroup_mean_fillna(df, 'target', 'group')
    """
    if newcolumn == False:
        df[target] = df.groupby(grouper)[target].transform(lambda x: x.fillna(x.mean()))
    else:
        df[newcolumn] = df.groupby(grouper)[target].transform(lambda x: x.fillna(x.mean()))

### Plots
def scatter1(df, xplot, yplot):
    """
    Create a scatter plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.

    Example:
        >>> scatter1(df, 'column1', 'column2')
    """
    fig, ax = plt.subplots()
    ax.scatter(x=df[xplot], y=df[yplot])
    plt.ylabel(yplot, fontsize=13)
    plt.xlabel(xplot, fontsize=13)
    plt.show()

def stripplot1(df, xplot, yplot, figsize=(13,7), hueplot=False):
    """
    Create a strip plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.
        figsize (tuple): The size of the figure. Defaults to (13,7).
        hueplot (bool or str): If False, create a basic strip plot; if a string, use it as the column name for hue grouping. Defaults to False.

    Example:
        >>> stripplot1(df, 'column1', 'column2', figsize=(10,5), hueplot='category')
    """
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)
    if hueplot == False:
        sns.stripplot(x=xplot, y=yplot, data=df[[xplot, yplot]])
    else:
        sns.swarmplot(x=xplot, y=yplot, hue=hueplot, data=df[[xplot, yplot, hueplot]])

def swarmplot1(df, xplot, yplot, figsize=(13,7), hueplot=False):
    """
    Create a swarm plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.
        figsize (tuple): The size of the figure. Defaults to (13,7).
        hueplot (bool or str): If False, create a basic swarm plot; if a string, use it as the column name for hue grouping. Defaults to False.

    Example:
        >>> swarmplot1(df, 'column1', 'column2', figsize=(10,5), hueplot='category')
    """
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)
    if hueplot == False:
        sns.swarmplot(x=xplot, y=yplot, data=df[[xplot, yplot]])
    else:
        sns.swarmplot(x=xplot, y=yplot, hue=hueplot, data=df[[xplot, yplot, hueplot]])

def boxplot1(df, xplot, yplot, figsize=(13,7), hueplot=False):
    """
    Create a box plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.
        figsize (tuple): The size of the figure. Defaults to (13,7).
        hueplot (bool or str): If False, create a basic box plot; if a string, use it as the column name for hue grouping. Defaults to False.

    Example:
        >>> boxplot1(df, 'column1', 'column2', figsize=(10,5), hueplot='category')
    """
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)
    if hueplot == False:
        sns.boxplot(x=xplot, y=yplot, data=df[[xplot, yplot]])
    else:
        sns.boxplot(x=xplot, y=yplot, hue=hueplot, data=df[[xplot, yplot, hueplot]])

def barplot1(df, xplot, yplot, figsize=(13,7), hueplot=False):
    """
    Create a bar plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.
        figsize (tuple): The size of the figure. Defaults to (13,7).
        hueplot (bool or str): If False, create a basic bar plot; if a string, use it as the column name for hue grouping. Defaults to False.

    Example:
        >>> barplot1(df, 'column1', 'column2', figsize=(10,5), hueplot='category')
    """
    sns.set(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)
    if hueplot == False:
        sns.barplot(x=xplot, y=yplot, data=df[[xplot, yplot]])
    else:
        sns.barplot(x=xplot, y=yplot, hue=hueplot, data=df[[xplot, yplot, hueplot]])

def factorplot1(df, xplot, yplot, colplot, size_aspect=[7,.5], hueplot=False):
    """
    Create a factor plot of two columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.
        colplot (str): The name of the column to create subplots based on.
        size_aspect (list): The size and aspect ratio of the figure. Defaults to [7, .5].
        hueplot (bool or str): If False, create a basic factor plot; if a string, use it as the column name for hue grouping. Defaults to False.

    Example:
        >>> factorplot1(df, 'column1', 'column2', 'column3', size_aspect=[10, 1], hueplot='category')
    """
    sns.set_context("paper", font_scale=1.5)
    if hueplot == False:
        sns.factorplot(x=xplot, y=yplot, col=colplot, data=df, kind="box", size=size_aspect[0], aspect=size_aspect[1])
    else:
        sns.factorplot(x=xplot, y=yplot, hue=hueplot, col=colplot, data=df, kind="box", size=size_aspect[0], aspect=size_aspect[1])

def snsjointplot(df, xplot, yplot):
    """
    Create a joint plot of two columns in a DataFrame using Seaborn.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot on the x-axis.
        yplot (str): The name of the column to plot on the y-axis.

    Example:
        >>> snsjointplot(df, 'column1', 'column2')
    """
    sns.set(style="darkgrid", color_codes=True)
    g = sns.jointplot(df[xplot], df[yplot], kind="reg", xlim=(df[xplot].min()+0.01, df[xplot].max()), ylim=(df[yplot].min()+0.01, df[yplot].max()), color="r", size=7)

def distplot1(df, xplot):
    """
    Create a distribution plot of a column in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        xplot (str): The name of the column to plot.

    Example:
        >>> distplot1(df, 'column1')
    """
    sns.distplot(df[xplot], rug=True, rug_kws={"color": "g"}, kde_kws={"color": "k", "lw": 3, "label": "KDE"}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})

### Distribution Normalizing
def mass_skew_sqrt(df):
    """
    Apply the square root transformation to columns with high skewness in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Example:
        >>> mass_skew_sqrt(df)
    """
    highskew_columns = df.skew().where(df.skew().apply(np.absolute) > 1.25).dropna().index.tolist()
    for i in highskew_columns:
        df[i] = np.sqrt(df[i])

def mass_skew_log(df):
    """
    Apply the natural logarithm transformation to columns with high skewness in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Example:
        >>> mass_skew_log(df)
    """
    highskew_columns = df.skew().where(df.skew().apply(np.absolute) > 1.25).dropna().index.tolist()
    for i in highskew_columns:
        df[i] = np.log(df[i])

def mass_skew_transform(df):
    """
    Apply different transformations to columns with high skewness or zero minimum values in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.

    Example:
        >>> mass_skew_transform(df)
    """
    zeromin_columns = (df.min() == 0).sort_values().tail(6).index.tolist()
    for i in zeromin_columns:
        df[i] = np.sqrt(df[i])
    highskew_negative_columns = df.skew().where(df.skew() < -1.25).dropna().index.tolist()
    for i in highskew_negative_columns:
        df[i] = np.sqrt(df[i])
    highskew_positive_columns = df.skew().where(df.skew() > 1.25).dropna().index.tolist()
    for i in highskew_positive_columns:
        df[i] = np.log(df[i])

def run_class_model(model, X_tr, y_tr, X_te, y_te, df_test=False, params=False):
    """
    Train and evaluate a classification model using the given training and test data.

    Parameters:
        model (class): The classification model class.
        X_tr (array-like): The training feature data.
        y_tr (array-like): The training target data.
        X_te (array-like): The test feature data.
        y_te (array-like): The test target data.
        df_test (DataFrame): The test DataFrame for prediction. Defaults to False.
        params (dict): The parameters for the model. Defaults to False.

    Returns:
        list: A list containing the predicted values on the test data, predicted values on the test DataFrame, fitted model, and model object.

    Example:
        >>> model_class_fit_pred, model_class_fit_pred2, model_class_fit, model_class = run_class_model(model, X_train, y_train, X_test, y_test, df_test=df_test, params=params)
    """
    if params == False:
        model_class = model()
    else:
        model_class = model(**params)

    model_class_fit = model_class.fit(X_tr, y_tr)
    model_class_fit_pred = model_class_fit.predict(X_te)
    if df_test != False:
        model_class_fit_pred2 = model_class_fit.predict(df_test)
    else:
        model_class_fit_pred2 = df_test
    return [model_class_fit_pred, model_class_fit_pred2, model_class_fit, model_class]

def single_model_tuner(model_list, X_tr, y_tr, X_te, y_te, deviation_function=False, modeling_type="Classification", short_list=False):
    """
    Tune hyperparameters for a single classification or regression model and evaluate its performance.

    Parameters:
        model_list (list): The list containing the model and its parameters.
        X_tr (array-like): The training feature data.
        y_tr (array-like): The training target data.
        X_te (array-like): The test feature data.
        y_te (array-like): The test target data.
        deviation_function (bool): If True, use deviation function for model evaluation. Defaults to False.
        modeling_type (str): The type of modeling, either 'Classification' or 'Regression'. Defaults to "Classification".
        short_list (bool): If True, return a short list of top-performing parameter combinations. Defaults to False.

    Returns:
        DataFrame: A DataFrame containing the evaluation metrics and parameters for each model.

    Example:
        >>> tuned_params_df = single_model_tuner(model_list, X_train, y_train, X_test, y_test, deviation_function=False, modeling_type="Classification", short_list=False)
    """
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    output_list = []
    output_columns = ['ModelName', 'Deviation', 'Accuracy', 'R2', 'ROC AUC', 'RMSE', 'Params', 'Model']
    dummylist = []
    if short_list == False:
        for i in model_list[3]:
            if model_list[5] == modeling_type:
                model = model_list[0](**i)
                model_fit = model.fit(X_tr.values, y_tr.values)
                model_fit_pred = model_fit.predict(X_te)
                r2score_output = r2_score(y_te, model_fit_pred)
                roc_auc_output = roc_auc_score(y_te, model_fit_pred)
                rmse_output = np.sqrt(mean_squared_error(y_te, model_fit_pred))
                output_list.append([model_list[1], (np.abs(y_te - model_fit_pred).sum() / model_fit_pred.shape)[0], accuracy_score(y_test2, model_fit_pred), r2score_output, roc_auc_output, rmse_output, model_list[5], model_fit])
            else:
                dummylist.append(i)
                return dummylist
    else:
        for i in model_list[6]:
            if model_list[5] == modeling_type:
                model = model_list[0](**i)
                model_fit = model.fit(X_tr, y_tr)
                model_fit_pred = model_fit.predict(X_te)
                r2score_output = r2_score(y_te, model_fit_pred)
                roc_auc_output = roc_auc_score(y_te, model_fit_pred)
                rmse_output = np.sqrt(mean_squared_error(y_te, model_fit_pred))
                output_list.append([model_list[1], (np.abs(y_te - model_fit_pred).sum() / model_fit_pred.shape)[0], accuracy_score(y_test2, model_fit_pred), r2score_output, roc_auc_output, rmse_output, model_list[5], model.get_params(), model])
            else:
                dummylist.append(i)
                return dummylist

    tuned_df = pd.DataFrame(output_list, columns=output_columns)
    model_list[4] = tuned_df.sort_values(by="RMSE", ascending=False).sort_values(by='Deviation').reset_index().head(1)['Params'].values[0]
    fulllistcsv = 'tuning_table_' + model_list[1].strip() + '.csv'
    shortlistcsv = 'tuning_table_' + model_list[1].strip() + '_short.csv'
    if short_list == False:
        tuned_df.to_csv(fulllistcsv)
        print(model_list[1] + ' Done. Saved to: ' + fulllistcsv)
    else:
        tuned_df.to_csv(shortlistcsv)
        print(model_list[1] + ' Done. Saved to: ' + shortlistcsv)

    return tuned_df

def multi_model_tuner(model_list, X_tr, y_tr, X_te, y_te, short_list=True, modeling_type="Classification", threshold=['Deviation', 0.20]):
    """
    Tune hyperparameters for multiple classification or regression models and evaluate their performance.

    Parameters:
        model_list (list): The list containing multiple models and their parameters.
        X_tr (array-like): The training feature data.
        y_tr (array-like): The training target data.
        X_te (array-like): The test feature data.
        y_te (array-like): The test target data.
        short_list (bool): If True, return a short list of top-performing parameter combinations. Defaults to True.
        modeling_type (str): The type of modeling, either 'Classification' or 'Regression'. Defaults to "Classification".
        threshold (list): The threshold used for selecting top-performing models based on the specified deviation metric. Defaults to ['Deviation', 0.20].

    Returns:
        tuple: A tuple containing three elements - a DataFrame of all parameter combinations and evaluation metrics, a DataFrame of predictions from the top-performing models, and a list of models with top-performing parameters already loaded.

    Example:
        >>> multi_tuned_params, multi_tuned_pred, multi_tuned_prefit = multi_model_tuner(model_list, X_train, y_train, X_test, y_test, short_list=True, modeling_type="Classification", threshold=['Deviation', 0.20])
    """
    output_columns = ['ModelName', 'Deviation', 'Accuracy', 'R2', 'ROC AUC', 'RMSE', 'Params', 'Model']
    multi_tuned_params = single_model_tuner(model_list[0], X_tr, y_tr, X_te, y_te, short_list=short_list, modeling_type=modeling_type)

    for i in model_list[1:]:
        multi_tuned_params = multi_tuned_params.append(single_model_tuner(i, X_tr, y_tr, X_te, y_te, short_list=short_list, modeling_type=modeling_type))

    if short_list == True:
        multi_tuned_params.to_csv('multi_tuned_params_short.csv')
        print("Multimodel Parameters Calculated. Saved to CSV: multi_tuned_params_short.csv")
    else:
        multi_tuned_params.to_csv('multi_tuned_params.csv')
        print("Multimodel Parameters Calculated. Saved to CSV: multi_tuned_params.csv")

    print('\n...Moving on to Producing the combined prediction\n')
    multi_tuned_prefit = [
        [i,
         multi_tuned_params.loc[(multi_tuned_params['ModelName'] == i) & (multi_tuned_params[threshold[0]] < 0.20)].sort_values(by='RMSE')['Model'].head(20).tolist()
        ] for i in multi_tuned_params['ModelName'].value_counts().index.tolist()]
    columnnames = []
    modelss = []
    k = 0
    for i in multi_tuned_prefit:
        for j in i[1]:
            k += 1
            columnnames.append(i[0] + str(k))
            modelss.append(pd.Series(j.fit(X_tr, y_tr).predict(X_te), index=X_te.index.values, name=i[0] + str(k)))

    multi_tuned_pred = pd.DataFrame(pd.DataFrame(modelss).T, index=X_te.index.values)
    multi_tuned_pred.columns = columnnames
    multi_tuned_pred['Mode'] = stats.mode([multi_tuned_pred[i] for i in multi_tuned_pred]).mode[0]
    multi_tuned_pred['Mean'] = np.mean([multi_tuned_pred[i] for i in multi_tuned_pred.drop('Mode', axis=1)])
    multi_tuned_pred['Y_True'] = y_te
    multi_tuned_pred['Mode_Error'] = np.abs(multi_tuned_pred['Y_True'] - multi_tuned_pred['Mode'])
    multi_tuned_pred['Mean_Error'] = np.abs(y_te - multi_tuned_pred['Mean'])

    multi_tuned_pred.to_csv('multi_tuned_pred.csv')
    print("Multimodel Parameters Calculated. Saved to CSV: multi_tuned_pred.csv")
    if modeling_type == "Classification":
        print('Mode of Pred RMSE:', np.sqrt(mean_squared_error(y_test2, multi_tuned_pred['Mode'])))
        print('Mode of Pred Deviation:', (multi_tuned_pred['Mode_Error'].sum() / multi_tuned_pred['Mode_Error'].shape))
        print('Error Count:', multi_tuned_pred.Mode_Error.sum())
    else:
        print('Mean of Pred RMSE:', np.sqrt(mean_squared_error(y_test2, multi_tuned_pred['Mean'])))
        print('Mean of Pred Deviation:', (multi_tuned_pred['Mode_Error'].sum() / multi_tuned_pred['Mode_Error'].shape))

    return multi_tuned_params, multi_tuned_pred, multi_tuned_prefit



classifiers = [   
    [AdaBoostClassifier,'AdaBoostClassifier',
     {'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
             max_features='log2', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=2, min_samples_split=6,
             min_weight_fraction_leaf=0.0, presort=False, random_state=4,
             splitter='best'),
 'learning_rate': 0.1,
 'n_estimators': 25,
 'random_state': 11},
     ParameterGrid({
    'learning_rate': [0.1,0.2,0.5,0.7], 
    'n_estimators': [10,25,50,75,100,200], 
    'random_state': [1,3,7,11,15], 
    'base_estimator': [DecisionTreeClassifier(**i) for i in ParameterGrid({
        'max_depth': [5,6,7,8],
        'max_features': ['log2'], 
        'min_samples_leaf': [2,3,4,5,6], 
        'min_samples_split': [2,3,4,5,6,7],
        'random_state': [1, 2,3,4]
     })]}), 
     {}, 
     'Classification',
     ParameterGrid({'n_estimators': [25,100,200],'learning_rate': [0.1,0.2], 'random_state':[3,7,11] })
],
    [
        XGBClassifier, 
        
        "XGBClassifier",
        
        {'colsample_bytree': 0.7, 'gamma': 1, 'learning_rate': 0.002, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 500, 'subsample': 0.9},
        
        ParameterGrid({
    'n_estimators': [500,600], 'max_depth': [3,5,6,7],
    'learning_rate': [0.002], 'colsample_bytree': [0.4603,0.5,0.7,0.8,1],
    'gamma': [0.0468, 1,2,3], 'min_child_weight': [1,2,3],
    'subsample': [0.5,0.7,0.9,1]}),
        
        {}, 
        
        'Classification',
        
        ParameterGrid({'n_estimators': [500],'max_depth': [7],
    'learning_rate': [0.002], 'colsample_bytree': [0.4603,0.7,1],
    'min_child_weight': [1,2,3]})],
    
    [RandomForestClassifier, 
     
     'RandomForestClassifier',
     
     {
      'class_weight': None,
      'criterion': 'gini',
      'max_depth': 7,
      'max_features': 'log2',
      'max_leaf_nodes': None,
      'min_impurity_decrease': 0.0,
      'min_impurity_split': None,
      'min_samples_leaf': 1,
      'min_samples_split': 13,
      'min_weight_fraction_leaf': 0.5,
      'n_estimators': 1000,
      'n_jobs': -1,
      'oob_score': False,
      'random_state': 10,
      'verbose': 0,
      'warm_start': False,
      'bootstrap': True,
      }, 
     
     ParameterGrid({
         'criterion': ['entropy','gini'],
         'max_depth': [2,3,4,5,6,7],
         'max_features': ['auto','sqrt','log2'],
         'min_samples_split': [2,4,7,10,13],
         'min_weight_fraction_leaf': [0,0.5],
         'n_estimators': [25,50, 100, 500, 1000],
         'n_jobs': [-1],
         'random_state': [None,2, 4, 7, 10]}), 
     
     {'learning_rate': 0.02,
  'max_depth': 4,
  'max_features': 0.1,
  'min_samples_leaf': 50,
  'min_samples_split': 13,
  'n_estimators': 500}, 
     'Classification',
     ParameterGrid({
         'max_depth': [3,5,7],
         'min_weight_fraction_leaf': [0,0.5],
         'n_estimators': [100, 500, 1000],
         'n_jobs': [-1],
         'random_state': [None,2, 4]}), ],
    [CatBoostClassifier,
     'CatBoost', 
     {'verbose':False, 'iterations':50, 
      'depth':3, 'learning_rate':0.1,'leaf_estimation_iterations': 10, 'eval_metric': 'Accuracy'}, 
     ParameterGrid({
         'verbose':[False],
         'depth':[3,1,2,6,4,5,7,8,9,10],
         'iterations':[250,100,500,1000],
         'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
         'l2_leaf_reg':[3,1,5,10,100],
         'border_count':[32,5,10,20,50,100,200],
         'loss_function': ['Logloss', 'CrossEntropy'],
          'thread_count':4}), {}, 'Classification',
    ParameterGrid({
        'verbose':[False],
         'depth':[3,5,7],
         'iterations':[100],
         'learning_rate':[0.1,0.3], 
         'l2_leaf_reg':[3,5,10],
         'loss_function': ['Logloss', 'CrossEntropy'],}),
    ],
    
    [LGBMClassifier,'LightGBM',
         {'boosting_type': 'gbdt',
         'class_weight': None,
         'colsample_bytree': 1.0,
         'learning_rate': 0.1,
         'max_depth': -1,
         'min_child_samples': 20,
         'min_child_weight': 0.001,
         'min_split_gain': 0.0,
         'n_estimators': 100,
         'n_jobs': -1,
         'num_leaves': 31,
         'objective': None,
         'random_state': None,
         'reg_alpha': 0.0,
         'reg_lambda': 0.0,
         'silent': True,
         'subsample': 1.0,
         'subsample_for_bin': 200000,
         'subsample_freq': 0},
     ParameterGrid({
    'learning_rate': [0.005],
    'n_estimators': [40],
    'num_leaves': [6,8,12,16],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }), {},'Classification',ParameterGrid({'n_estimators':[200, 600, 80], 'num_leaves':[20,60,10]})],
    [GradientBoostingClassifier,'SK Gradient Boosting',
     {'criterion': 'friedman_mse',
      'init': None,
      'learning_rate': 0.1,
      'loss': 'deviance',
      'max_depth': 3,
      'max_features': None,
      'max_leaf_nodes': None,
      'min_impurity_decrease': 0.0,
      'min_impurity_split': None,
      'min_samples_leaf': 1,
      'min_samples_split': 2,
      'min_weight_fraction_leaf': 0.0,
      'n_estimators': 100,
      'presort': 'auto',
      'random_state': None,
      'subsample': 1.0,
      'verbose': 0,
      'warm_start': False}, 
     ParameterGrid({
         'learning_rate': [0.1, 0.05, 0.2, 0.5],
         'max_depth': [3, 4, 6, 8],
         'min_samples_leaf': [20, 50,100,150],
         'n_estimators': [100,500, 700, 1000],
         'max_features': [1.0, 0.3, 0.1],
         'min_samples_split': [2,4,6,8,10,13]     
     }), 
     {}, 
     'Classification', 
     ParameterGrid({
         'learning_rate': [0.1, 0.5],
         'max_depth': [4, 6, 8],
         'n_estimators': [100,500, 1000],
         'max_features': [1.0, 0.3, 0.1],
     })],
    [SVC,'SKLearn SVC',
     {'C': 1.0,
      'cache_size': 200,
      'class_weight': None,
      'coef0': 0.0,
      'decision_function_shape': 'ovr',
      'degree': 3,
      'gamma': 'auto',
      'kernel': 'rbf',
      'max_iter': -1,
      'probability': False,
      'random_state': None,
      'shrinking': True,
      'tol': 0.001,
      'verbose': False}, 
     ParameterGrid({
         'C':[0.001, 0.01, 0.1, 1, 10],
         'gamma':[0.001, 0.01, 0.1, 1],
         'coef0':[2, 5]
     }), 
     {}, 
     "Classification",
     ParameterGrid({
         'C':[0.01, 0.1],
         'gamma':[0.01, 0.1],
         'coef0':[2.0, 5.0, 9.0],
     })],
    [KNeighborsClassifier,'SK KNNeighbors',
     {'algorithm': 'auto',
      'leaf_size': 30,
      'metric': 'minkowski',
      'metric_params': None,
      'n_jobs': 1,
      'n_neighbors': 5,
      'p': 2,
      'weights': 'uniform'},
     ParameterGrid({
         'n_neighbors':np.arange(20)+1, 
         'weights':['uniform','distance'],
         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
     }),
     {},
     'Classification',
     ParameterGrid({
         'n_neighbors':np.arange(20)+1, 
         'weights':['uniform','distance'],
         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']})
    ]
]
    
 # classifer_schema = ['Model', 'Name', 'Base Params', 'ParameterGrid(List)', 'ParameterGrid Resulting Params', 'Regression', 'Short ParamGrid']
regressors = [
        [
                LinearRegression, 
                'LinearRegression', 
                {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False}, 
                ParameterGrid({}), 
                {}, 
                'Regression', 
                ParameterGrid({}),
        ],

         [
                 RandomForestRegressor, 
                 'RandomForestRegressor', 
                 {'bootstrap': True,
                  'criterion': 'mse',
                  'max_depth': None,
                  'max_features': 'auto',
                  'max_leaf_nodes': None,
                  'min_impurity_decrease': 0.0,
                  'min_impurity_split': None,
                  'min_samples_leaf': 1,
                  'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0,
                  'n_estimators': 10,
                  'n_jobs': 1,
                  'oob_score': False,
                  'random_state': None,
                  'verbose': 0,
                  'warm_start': False}, 
                 ParameterGrid({
                         'n_estimators': [500, 650, 700, 750, 800, 1000], 
                         'max_depth': [None, 1, 2, 3, 10, 11, 12], 
                         'min_samples_split': [2, 3, 4, 5, 6],
                         'min_samples_leaf' : [2, 3, 4, 5, 6],
                         "criterion": ["gini", "entropy", 'mse', 'mae'],
                         "bootstrap": [True, False],
                         }), 
                 ParameterGrid({
                         'n_estimators': [500, 1000], 
                         'max_depth': [3, 6, 12], 
                         'min_samples_split': [2, 3, 4],
                         'min_samples_leaf' : [2, 3],
                         }), 
                 'Regression', 
                 'Short ParamGrid'
        ],
        [
                GradientBoostingRegressor, 
                'GradientBoostingRegressor', 
                {'alpha': 0.9,
                 'criterion': 'friedman_mse',
                 'init': None,
                 'learning_rate': 0.1,
                 'loss': 'ls',
                 'max_depth': 3,
                 'max_features': None,
                 'max_leaf_nodes': None,
                 'min_impurity_decrease': 0.0,
                 'min_impurity_split': None,
                 'min_samples_leaf': 1,
                 'min_samples_split': 2,
                 'min_weight_fraction_leaf': 0.0,
                 'n_estimators': 100,
                 'presort': 'auto',
                 'random_state': None,
                 'subsample': 1.0},
                ParameterGrid({
                        'n_estimators': [100, 300, 500], 
                        'max_depth': [3,4,5],
                        'learning_rate': [0.1,0.05,0.02],
                        'min_samples_leaf':[3, 5, 9, 17],
                        'max_features':[0.1, 0.3, 1.0] 
                        }), 
                'ParameterGrid Resulting Params', 
                'Regression', 
                ParameterGrid({
                        'max_depth': [3,5],
                        'n_estimators':[100,500], 
                        'learning_rate': [0.1,0.5],
                        'min_samples_leaf':[3],
                        'max_features':[1.0] 
                        })
        ],
        [
                CatBoostRegressor, 
                'CatBoostRegressor', 
                'Base Params', 
                ParameterGrid({
                        'depth': [4, 6, 8],
                        'random_seed' : [400, 100, 200],
                        'learning_rate' : [0.01, 0.03, 0.1]}), 
                'ParameterGrid Resulting Params', 
                'Regression', 
                'Short ParamGrid'
        ],
        [
                XGBRegressor, 
                'XGBRegressor', 
                'Base Params', 
                'ParameterGrid(List)', 
                'ParameterGrid Resulting Params', 
                'Regression', 
                'Short ParamGrid'
        ],
        [
                LGBMRegressor, 
                'LGBMRegressor', 
                'Base Params', 
                'ParameterGrid(List)', 
                'ParameterGrid Resulting Params', 
                'Regression', 
                'Short ParamGrid'
        ],
        [
                ElasticNet, 
                'ElasticNet', 
                'Base Params', 
                'ParameterGrid(List)', 
                'ParameterGrid Resulting Params', 
                'Regression', 
                'Short ParamGrid'
        ],
        [
                Lasso, 
                'ElasticNet', 
                'Base Params', 
                'ParameterGrid(List)', 
                'ParameterGrid Resulting Params', 
                'Regression', 
                'Short ParamGrid'
        ]
        ]
clusters = []
reducers = []
untuned_models = [classifiers,regressors,clusters,reducers]





print(len(regressors))


