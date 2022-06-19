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

#the function cleandata addresses general problems with unclean data
#It also fills missing values with most frequent value in each column
def cleandata(data):
	print("Cleaning...")
	print('\n')
    #This condition is used to address problems with importing data which was exported using pandas
    #We remove the first column which contains indices and is unnamed
	if data.columns[0]=='Unnamed: 0':
        #iloc function is used for indexing
	    data=data.iloc[:,1:]

    #print first 5 rows of the dataset
	print("First 5 rows of data:")
	print(data.head())
	print('\n')

    #The block below gives quick summary of data
	print("Summary of Dataset")
    #Size refers to total number of values in dataset
	print("Size: ",data.size)
    #Shape gives number of rows and columns of dataset
	print("Shape: ",data.shape)
    #Dim gives the dimension of the dataset
	print("Dimensions: ",data.ndim)
	print('\n')

    #The below codeblock is used to drop all rows which have all values as null
	data.dropna(axis=1,
	            inplace=True,
	            how='all')
    #We calculate and print percentage of missing values in each column
	percentage_missing=data.isnull().sum()*100/data.shape[0]
	print("Percentage of Missing Values: ")
	print(percentage_missing)
	print('\n')
	print('\n')

    #In the codeblock below, we drop all those columns which have 50% or more missing values
	cols_to_drop=[]
	print("Columns with more than 50% missing values are:")
	print("Column Number  Column Name")
	for i in range(len(percentage_missing)):
	    if percentage_missing[i] > 50.00:
            #We create a list of columns to drop
	        cols_to_drop.append(data.columns[i])
            #The below print line uses formatting
	        print(f"{i}             {data.columns[i]}")
	print('\n')

    #The code below drops all the columns having more than 50% missing values
	print('Dropping...')
	data.drop(cols_to_drop,axis=1,inplace=True)
	print("Dropped.")
	print('\n')

    #We print the summary of dataset after dropping
	print("Summary of Dataset After Dropping")
	print("Size: ",data.size)
	print("Shape: ",data.shape)
	print("Dimensions: ",data.ndim)
	print('\n')

    #We calculate percentage of missing values again,and also drop rows which have all null values after the above operation
	data.dropna(axis=1,inplace=True,how='all')
	percentage_missing=data.isnull().sum()*100/data.shape[0]
	print("Percentage of Missing Values After Dropping Rows with All Null Values: ")
	print(percentage_missing)
	print('\n')
	print('\n')

    #The below codeblock is specific to our data hence it is wrapped in try and except blocks
	print("Dropping ID and Dates")
	try:
        #For our data, we drop these 3 columns
	    data.drop(['ID','Start Date (UTC)','Submit Date (UTC)'],
	              axis=1,
	              inplace=True)
	except:
        #For other datasets
	    print("One or More Columns not found")
	    print('\n')
    #We print the summary of dataset after dropping
	print("Dropped")
	print('\n')
	print("Summary of Dataset After Dropping 'ID','Start Date (UTC)','Submit Date (UTC)'")
	print("Size: ",data.size)
	print("Shape: ",data.shape)
	print("Dimensions: ",data.ndim)
	print('\n')

    #The counter function is used to count freqency of values
	from collections import Counter
    #Pprint is used to print data in an organized way
	from pprint import pprint
    #value_counts holds frequency of values for all columns
	value_counts=dict()
    #for each column, we build a frequency histogram
	for column in data.columns:
        #We drop null values (temporary)
	    temp=data[column].dropna()
        #We build the frequency histogram and put it to value)counts[column]
	    value_counts[column]=len(dict(Counter(temp)
                                     )
                                )
	print("Frequency of Values:")
	pprint(value_counts)
	print('\n')
    #We print all those columns which have number of values greater than 10
	print("Columns having more than 10 unique values:")
	for k,v in value_counts.items():
	    if v>10:
	        print(k,v)
	print('\n')
    #Calculate percentage of missing values
	print("Percentage of Missing Values After Dropping 'ID','Start Date (UTC)','Submit Date (UTC)': ")
	percentage_missing=data.isnull().sum()*100/data.shape[0]
	print(percentage_missing)
	print('\n')
	print('\n')

    #We put all the columns which have missing values in a list
	missing_cols=[]
	for column in data.columns:
        #If percentage of missing values is greater than 0, we append it to the list
	    if percentage_missing[column]>0:
	        missing_cols.append(column)
    #Print the number of columns which have missing values
	print("Number of columns having missing values",len(missing_cols))
    #The next thing we need to do is to count the frequency of values 
    #in columns having missing values to replace it by most frequent value
    #We create a dictionary to hold frequency values for each column
	print("Calculating Most Frequent Values in Each Column...")
	print('\n')
	total_counts={}
    #We use counter module for counting
	from collections import Counter
    #We are using isnan function to check if a vaue is null
	from math import isnan
    #The below loop builds frequency histogram
	for column in missing_cols:
        #counts holds frequency values for the current column
	    counts=dict(Counter(data[column]))
        #We remove frequency values for nan
	    counts={k: counts[k] for k in counts if isinstance(k,str) or not isnan(k)}
        #We sort the frequenct histogram in descending order of frequency
	    ordered_counts=sorted(counts.items(),
	                          key=lambda x: x[1],
	                          reverse=True)
        #We append the sorted values to total_counts 
	    total_counts[column]=ordered_counts

    #We are now going to make a dictionary containing most frequent value for each column
	most_freq_values=dict()
    #Since we have sorted the frequencies, our value is the first tuple and key for each column
	for column in missing_cols:
        #We build dictionary of most frequent values
	    most_freq_values[column]=total_counts[column][0][0]
    #Finally, we fill all missing values with the most frequent value
	print("Done.")
	print('\n')
	print("Most Frequent Values in Each Column:")
	pprint(most_freq_values)
	print('\n')

	print("Filling Missing Values with Most Frequent Values...")
	print('\n')
	for column in missing_cols:
	    data[column].fillna(most_freq_values[column],inplace=True)
    #We print percentage of missing values after cleaning
	print("Done.")
	print("\n")
	print("Percentage of Missing Values After Cleaning: ")
	percentage_missing=data.isnull().sum()*100/data.shape[0]
	print(percentage_missing)
	print('\n')
	print('\n')
	print("Cleaning Completed.")
	print("\n")
    #We Return cleaned data
	return data

def preprocess():
	#We call get data function to get data from commandline/default and store it in variable data
	print("Fetching Data...")
	print("\n")
	data=getdata()
	print("Done.")
	print("\n")
	#We call cleandata, and save the cleaned data in data
	data=cleandata(data)
	#We print the results after cleaning
	print("Final Results:")
	print(data.isnull().sum())
	#Export the cleaned data to the file cleaned.csv
	print("Saving Data to cleaned.csv..")
	print("\n")
	data.to_csv("cleaned.csv")
	print("Completed.")
	print("\n")
preprocess()