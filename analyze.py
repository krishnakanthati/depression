#!/usr/bin/env python
# coding: utf-8

# Documentation and Annotations are written in comment lines
# This python script is an automated data cleaner written for our final year project 
# but it is supposed to work with any csv file.
# Project Name: Screening Depression Using Machine Learning
# Team: Pareekshith US Katti, Ganesh Manu Mahesh Kashyap, Sanjay R Rao, Jitendra Kumar Mahto

#The sys module is used in the project to read filename from commandline
import sys

#Pandas package is used for reading and cleaning data
import pandas as pd

#The function getdata gets data from commandline, if no filname is given, it takes default argument
def getdata():
	try:
		#get filename from commandline
		filename=sys.argv[1]
		#read data from file
		data=pd.read_csv(filename)
		print("Filename is,",filename)
		print('\n')
		#return data to a variable
		return data
	except:
		#if no filename is given in the arguments, use 2018.csv
		data=pd.read_csv('2018.csv')
		#return data to a variable
		return data

#This function removes anamolies in numerical data
def helper_function_anamolies(num):
    try:
        #Check if it is numerical
        num = pd.to_numeric(num)
    except:
        import numpy as np
        #Replace anamolies with nan
        num = np.nan
    return num

#This function gives basic dataset information
def get_dataset_info(df):
    info={}
    #Number of Features/Columns
    info['Features']=float(df.shape[1])
    #Number of Rows
    info['Observations']=float(df.shape[0])
    #Number of Missing Values
    info['Missing']=float(sum(df.isna().sum()))
    #Total memory usage
    info['Memory_Usage']=str(sum(df.memory_usage())/1024)+' kB'
    #Number of Possible Numerical Features
    info['No_of_Numerical_Features']=len(df.select_dtypes(['int64','float64']).columns)
    #Number of Possible Categorical Features
    info['No_of_Categorical_Features']=len(df.columns)-len(df.select_dtypes(['int64','float64']).columns)
    print("Dataset Information:")
    for k,v in info.items():
        print(k," : ",v)
    print('\n')

#This function gives information for each variable
def get_variable_info(df):
    info={}
    from collections import Counter
    for i in df.describe().columns:
        #Statistical Info
        info[i]=dict(df.describe()[i])
        #Missing Values Info for variable
        info[i]['missing']=float(df[i].isna().sum())
        #Datatype of the column
        info[i]['datatype']=str(df[i].dtype)
        #Correlation
        info[i]['correlation']=dict(df.corr()[i])
        #Possible type of feature
        info[i]['type_of_feature']='Numerical'
    for i in df.columns:
        if i not in df.describe().columns:
            df[i]=df[i].apply(str)
            info[i]=dict(Counter(df[i]))
            #Datatype of the column
            info[i]['datatype']=str(df[i].dtype)
            #Possible type of feature
            info[i]['type_of_feature']='Categorical'
    print("Variable Information:")
    for k,v in info.items():
        print(k," : ")
        for key,val in v.items():
            print(key," : ",val)
        print('\n')
    print('\n')

#This function suggests a method to handle missing values
def select_method(df):
    missing=df.isnull().sum()/df.shape[0]
    if missing==0.0:
        #No Missing Values
        return 'none'
    elif missing<=0.08:
        #Less than 8%
        return 'dropped'
    elif missing>0.08 and missing<=0.20:
        #Less than 20% and greater than 8%
        return 'ffill or bfill'
    elif missing>0.20 and missing<=0.40:
        #Less than 40% and greater than 20%
        return "interpolate"
    else:
        #More than 40%
        return "dropped, more than 40% missing"

#This function gives missing value statistics    
def missing_values_report(df):
    info={}
    for i in df.columns:
        #Get percentage of missing values, total number of missing values and suggested filling method
        info[i]={"percentage_of_missing_values":float(df[i].isnull().sum()*100/df[i].shape[0]),
                 "total_number_of_missing_values":float(df[i].isnull().sum()),
                 "filling_method":select_method(df[i])}
    print("Missing Values Information:")
    for k,v in info.items():
        print(k," : ")
        for key,val in v.items():
            print(key," : ",val)
        print('\n')
    print('\n')

#This function gives details about outliers
def outlier_report(df):
    info={}
    import numpy as np
    for i in df.select_dtypes(include=['int64','float64']):
        temp=df[i].dropna()
        #Get Outlier threshold, total number of outliers and percentage of outliers
        info[i]={'outlier_threshold':float(np.percentile(temp,99)),
                 'total_number_of_outliers':float(temp[temp>=np.percentile(temp,99)].count()),
                 'percentage_of_outliers':float(temp[temp>=np.percentile(temp,99)].count()/temp.shape[0])
                 }
    print("Outlier Information:")
    for k,v in info.items():
        print(k," : ")
        for key,val in v.items():
            print(key," : ",val)
        print('\n')
    print('\n')

#This function acts as a wrapper function to all of the methods    
def report_data():
    try:
        df=getdata()
        #df[numeric] = df[numeric].apply(lambda x: x.map(lambda a: helper_function_anamolies(a)))
        get_dataset_info(df)
        missing_values_report(df)
        outlier_report(df)
        get_variable_info(df)
    except Exception as e:
        return "caught err "+str(e)
report_data()