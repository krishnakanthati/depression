#!/usr/bin/env python
# coding: utf-8

# Documentation and Annotations are written in comment lines
# This python script is an automated data cleaner written for our final year project
# but it is supposed to work with any csv file.
# Project Name: Screening Depression Using Machine Learning
# Team: Pareekshith US Katti, Ganesh Manu Mahesh Kashyap, Sanjay R Rao, Jitendra Kumar Mahto

# The sys module is used in the project to read filename from commandline
import sys

# Pandas is used for handling data and correlation analysis
import pandas as pd


# In Target Encoding, features are replaced with a blend of posterior probability of the target given particular categorical value
# and the prior probability of the target over all the training data.
from category_encoders import TargetEncoder


# Pickle library is used to save python objects
import pickle


# The function getdata gets data from commandline, if no filname is given, it takes default argument
def getdata():
    try:
        # get filename from commandline
        filename = sys.argv[1]
        # read data from file
        data = pd.read_csv(filename)
        print("Filename is,", filename)
        print('\n')
        # return data to a variable
        return data
    except:
        # if no filename is given in the arguments, use cleaned.csv
        data = pd.read_csv('cleaned.csv')
        # return data to a variable
        return data


# This function is used to clean the gender column since it had 57 unique values, a custom cleaning function works better
def genderclean(x):
    # Strip Whitespaces at starting and ending of each values
    x = x.strip()
    # Convert to lower case
    x = x.lower()
    # If female is in the value
    if "female" in x:
        return 0
    # If male is in the value
    if "male" in x:
        return 1
    # If neither
    else:
        return 2
    return x


# This function encodes depresssion levels. Target encoder needs target column to be specified, we need to encode target column
def depressionencode(x):
    # No has the lowest level
    if x == 'No':
        return 0
    # Yes has the highest level
    if x == 'Yes':
        return 2
    if x == 'Possibly':
        return 2
    # Maybe or Probably has middle level
    else:
        return 1


# This function returns correlation for target column
def getCorr(data, col):
    # Get correlation for target column
    print("Correlation for ", col)
    print(data.corr()[col])
    print('\n')
    print('\n')
    # Return correlation for target column
    return data.corr()[col]


# This function encodes the categorical data using target encoder
def TargetEncode(data, target):
    # Select all categorical columns
    data_to_encode = data.select_dtypes(include=['object'])
    print('Data to be encoded: ')
    cols = list(data_to_encode.columns)
    print(len(cols))
    cols = '\n'.join(cols)
    print(cols)
    print('\n')
    print('\n')
    # For each column, encode using target encoder
    cols = list(data_to_encode.columns)
    model = TargetEncoder().fit(X=data[cols], y=data[target])
    # File where the target encoding model is saved
    filename = "targetencodemodel.sav"
    # Open file in binary mode
    f = open(filename, 'wb')
    # Dump model to file
    pickle.dump(model, f)
    f.close()
    print("Model saved in ", filename)
    print("\n")
    print("\n")
    # Read Model from file
    # f=open(filename,'rb')
    # model1=pickle.load(f)
    # f.close()
    # print("Model Loaded")
    # print(model1)
    # print('\n')
    data[cols] = model.transform(X=data[cols])
    # Return encoded data
    return data


# This function returns top n variables with highest correlation with target variable
def get_top_corr(corr, num):
    try:
        # Sort values in descending order of correlation
        corr = corr.sort_values(ascending=False)
        print("Top correlated variables")
        print(corr[:num])
        print('\n')
        print('\n')
        # Return top n correlated variables
        return corr[:num]
    except:
        return None


# This function returns top n variables with lowest correlation with target variable
def get_least_corr(corr, num):
    try:
        # Sort values in ascending order of correlation
        corr = corr.sort_values(ascending=True)
        print("Least correlated variables")
        print(corr[:num])
        print('\n')
        print('\n')
        # Return n least correlated variables
        return corr[:num]
    except:
        return None


# This function uses random forest algorithm to determine feature importance
def get_feature_importance(data, target, num):
    import numpy as np
    np.random.seed(67)
    # y contains the target
    y = data[target]
    # x contains features
    x = data.drop(target, axis=1)
    # We split it into training and testing
    from sklearn.model_selection import train_test_split
    # train_test_split splits with 33% test set
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    # Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.ensemble import RandomForestClassifier
    # Fit with training data
    model = RandomForestClassifier(
        n_estimators=512, random_state=35).fit(xtrain, ytrain)
    # predict the test data
    pred = model.predict(xtest)

    # initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print(accuracy_score(ytest, pred))
    print('\n')
    print('\n')
    print(model.feature_importances_)
    # get feature importance and construct a dataframe
    features = pd.DataFrame()
    # Contains column names
    features['feature'] = data.drop([target], axis=1).columns
    # Contains feature importance score
    features['importance'] = model.feature_importances_
    # Sort by feature importance
    features.sort_values(by=['importance'], ascending=False, inplace=True)
    # Print sorted order according to feature importance
    print("Feature Importance: ")
    print(features)
    print('\n')
    print('\n')
    # get important columns
    cols = features['feature']
    # convert to list
    cols = list(cols)
    # return top n important features
    return cols[:num]

# Main function


def main():
    print("Fetching Data...")
    print("\n")
    # We call get data function to get data from commandline/default and store it in variable data
    data = getdata()
    print("Done.")
    print("\n")
    # If the data is exported from pandas
    if data.columns[0] == 'Unnamed: 0':
        data = data.iloc[:, 1:]
    # Initial head of data
    print("First 5 rows:")
    print('\n')
    print(data.head())
    print('\n')
    print('\n')
    # data type of each column
    types = data.dtypes
    print("Data Types:")
    print('\n')
    print(types)
    print('\n')
    print('\n')
    try:
        # Clean the gender column
        data['gender '] = data['gender '].apply(genderclean)
        print("After processing gender column:")
        print('\n')
        print(data['gender '].unique())
        print('\n')
        print('\n')
        # Encode target variable
        data['Depression'] = data['Depression'].apply(depressionencode)
        print("Depression levels after encoding:")
        print('\n')
        print(data['Depression'].unique())
        print("First 5 rows:")
        print('\n')
        print(data.head())
        print('\n')
        print('\n')
        temp = data.copy()
        print("TEMP")
        print(temp.head())
        data = TargetEncode(data, 'Depression')
    except:
        print("Not applicable")
    print("First 5 rows after target encoding:")
    print('\n')
    print(data.head())
    print('\n')
    print('\n')
    # Get correlation for Depression
    corr = getCorr(data, 'Depression')
    # Get top correlated features
    top_corr = get_top_corr(corr, 25)
    cols = top_corr.index
    cols = list(cols)
    # Select top 25 correlated features
    data = data[cols]
    print("Selected Columns after Correlation Analysis")
    print('\n')
    cols = '\n'.join(cols)
    print(cols)
    print('\n')
    print('\n')
    # Get top 15 important features
    features = get_feature_importance(data, 'Depression', 15)
    featurestr = '\n'.join(features)
    print("Selected Columns after Feature Importance")
    print('\n')
    print(featurestr)
    features.append('Depression')
    temp = temp[features]
    temp.to_csv('notencoded.csv')
    # If the data is exported from pandas
    print(temp.shape)
    if temp.columns[0] == 'Unnamed: 0':
        temp = temp.iloc[:, 1:]
    temp = TargetEncode(temp, 'Depression')
    print('Saved Target Encoding for 15 features')
    print(temp.shape)
    temp.to_csv('temp.csv')
    data = data[features]
    # Save the encoded data to a csv file
    data.to_csv("encoded.csv")


main()
