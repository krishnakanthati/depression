#!/usr/bin/env python
# coding: utf-8

# Documentation and Annotations are written in comment lines
# This python script is an automated data cleaner written for our final year project 
# but it is supposed to work with any csv file.
# Project Name: Screening Depression Using Machine Learning
# Team: Pareekshith US Katti, Ganesh Manu Mahesh Kashyap, Sanjay R Rao, Jitendra Kumar Mahto

#The sys module is used in the project to read filename from commandline
import sys

#Pandas is used for handling data and correlation analysis
import pandas as pd


#In Target Encoding, features are replaced with a blend of posterior probability of the target given particular categorical value
#and the prior probability of the target over all the training data.
from category_encoders import TargetEncoder


#Pickle library is used to save python objects
import pickle 

from sklearn.model_selection import RandomizedSearchCV


#The function getdata gets data from commandline, if no filname is given, it takes default argument
def getdata():
    try:
        #get filename from commandline
        filename=sys.argv[1]
        #read data from file
        data=pd.read_csv(filename)
        print("Filename is,",filename)
        print('\n')
        #If the data is exported from pandas
        if data.columns[0]=='Unnamed: 0':
            data=data.iloc[:,1:]
        #return data to a variable
        return data
    except:
        #if no filename is given in the arguments, use encoded.csv
        data=pd.read_csv('encoded.csv')
        #If the data is exported from pandas
        if data.columns[0]=='Unnamed: 0':
            data=data.iloc[:,1:]
        #return data to a variable
        return data

        
def RandomForest_Opt():
    import numpy as np
    np.random.seed(67)
    print("Random Forest Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Random Forest
    space4rf = {
    'max_depth': range(1,20),
    'n_estimators': [32,64,128,256,512],
    'criterion': ["gini", "entropy"],
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomizedSearchCV(estimator = RandomForestClassifier(), 
                           param_distributions = space4rf, 
                           n_iter = 50, 
                           cv = 3,
                           random_state=42,
                           verbose=2, 
                           n_jobs = -1)
    print("Random State: ",clf.random_state)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    print(model)
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="randomforest.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        #We split it into training and testing
        from sklearn.model_selection import train_test_split
        #train_test_split splits with 33% test set
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="randomforest.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"Random Forest"]


def DecisionTrees_Opt():
    import numpy as np
    np.random.seed(67)
    print("Decision Trees Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Decision Trees
    space4dt = {
    'max_depth': range(1,20),
    'criterion': ["gini", "entropy"],
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.tree import DecisionTreeClassifier
    clf=RandomizedSearchCV(estimator = DecisionTreeClassifier(), 
                           param_distributions = space4dt, 
                           n_iter = 50, 
                           cv = 3,
                           random_state=42,
                           verbose=2, 
                           n_jobs = -1)
    print("Random State: ",clf.random_state)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    print(model)
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="dt.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="dt.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"Decision Tree"]


def GradientBoost_Opt():
    import numpy as np
    np.random.seed(67)
    print("Gradient Boosting Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Gradient Boosting
    space4gb = {
    'max_depth': range(1,20),
    'n_estimators': [32,64,128,256,512],
    'learning_rate':[0.01,0.05,0.1,0.5]
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.ensemble import GradientBoostingClassifier
    clf=RandomizedSearchCV(estimator = GradientBoostingClassifier(), 
                           param_distributions = space4gb, 
                           n_iter = 50, 
                           cv = 3,
                           verbose=2, 
                           n_jobs = -1)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="gradientboosting.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="gradientboosting.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"Gradient Boost"]


def SVM_Opt():
    import numpy as np
    np.random.seed(67)
    print("SVM Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for SVM
    space4svc = {
    'kernel':['linear','rbf'],
    'gamma':[0.001, 0.01, 0.1, 1],
    'C':[0.001, 0.01, 0.1, 1, 10]
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.svm import SVC
    clf=RandomizedSearchCV(estimator = SVC(), 
                           param_distributions = space4svc, 
                           n_iter = 50, 
                           cv = 3,
                           verbose=2, 
                           n_jobs = -1)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="svm.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="svm.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"SVM"]


def AdaBoost_Opt():
    import numpy as np
    np.random.seed(67)
    print("Adaboost Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Adaboost
    space4ab = {
    'n_estimators': [32,64,128,256,512],
    'learning_rate':[0.01,0.05,0.1,0.5]
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.ensemble import AdaBoostClassifier
    clf=RandomizedSearchCV(estimator = AdaBoostClassifier(), 
                           param_distributions = space4ab, 
                           n_iter = 50, 
                           cv = 3,
                           random_state=42,
                           verbose=2, 
                           n_jobs = -1)
    print("Random State: ",clf.random_state)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    print(model)
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="adaboost.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        #We split it into training and testing
        from sklearn.model_selection import train_test_split
        #train_test_split splits with 33% test set
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="adaboost.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"AdaBoost"]


def XGBoost_Opt():
    import numpy as np
    np.random.seed(67)
    print("XGBoosting Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Gradient Boosting
    space4xgb = {
    'max_depth': range(1,20),
    'n_estimators': [32,64,128,256,512],
    'learning_rate':[0.01,0.05,0.1,0.5]
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from xgboost import XGBClassifier
    clf=RandomizedSearchCV(estimator = XGBClassifier(), 
                           param_distributions = space4xgb, 
                           n_iter = 50, 
                           cv = 3,
                           verbose=2, 
                           n_jobs = -1)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="xgboosting.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="xgboosting.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"XGBoost"]


def KNN_Opt():
    import numpy as np
    np.random.seed(67)
    print("KNN Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Gradient Boosting
    space4knn = {
    'n_neighbors':range(1,20),
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.neighbors import KNeighborsClassifier
    clf=RandomizedSearchCV(estimator = KNeighborsClassifier(), 
                           param_distributions = space4knn, 
                           n_iter = 50, 
                           cv = 3,
                           verbose=2, 
                           n_jobs = -1)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="knn.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="knn.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"KNN"]


def Bagging_Opt():
    import numpy as np
    np.random.seed(67)
    print("Bagging Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    #Hyper Parameters for Gradient Boosting
    space4bag = {
    'n_estimators': [32,64,128,256,512]
    }
    #Random Forest is used for getting feature importance, any tree based algorithm can be used
    from sklearn.ensemble import BaggingClassifier
    clf=RandomizedSearchCV(estimator = BaggingClassifier(), 
                           param_distributions = space4bag, 
                           n_iter = 50, 
                           cv = 3,
                           verbose=2, 
                           n_jobs = -1)
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    model=clf.best_estimator_
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="bagging.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="bagging.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(model,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"Bagging"]


def NaiveBayes_Opt():
    print("Naive Bayes Classifier")
    print('\n')
    print('\n')
    data=getdata()
    print(data.head())
    target='Depression'
    #y contains the target
    y=data[target]
    print(y)
    #x contains features
    x=data.drop(target,axis=1)
    print(x.head())
    #We split it into training and testing
    from sklearn.model_selection import train_test_split
    #train_test_split splits with 33% test set
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    print("Split complete")

    from sklearn.naive_bayes import MultinomialNB
    clf=MultinomialNB()
    #Fit with training data
    clf.fit(xtrain,ytrain)
    pred=clf.predict(xtest)
    # #initial accuracy score
    from sklearn.metrics import accuracy_score
    print("Accuracy Score with top correlated features: ")
    print("Accuracy",accuracy_score(ytest,pred))
    current_acc=accuracy_score(ytest,pred)
    print('\n')
    print('\n')
    from sklearn.metrics import classification_report
    print("Classification Report: ")
    print(classification_report(ytest,pred))
    print('\n')
    print('\n')
    try:
        #File where the model is saved
        filename="nb.sav"
        f=open(filename,'rb')
        prev_model=pickle.load(f)
        pred=prev_model.predict(xtest)
        prev_accuracy=accuracy_score(ytest,pred)
        from sklearn.metrics import classification_report
        print("Classification Report: ")
        print(classification_report(ytest,pred))
        print('\n')
        print('\n')
    except:
        prev_accuracy=0
    if current_acc>prev_accuracy:
        filename="nb.sav"
        #Open file in binary mode
        f=open(filename,'wb')
        #Dump model to file
        pickle.dump(clf,f)
        f.close()
        print("Model saved in ",filename)
        print("\n")
        print("\n")
    else:
        print("Previous Model is better")
        print(prev_accuracy)
    return [max(prev_accuracy,current_acc),"Naive Bayes"]


#Identifies Best Model and Gives Report of Model Building and Selection
def model_selection():
    #This list holds accuracy and name of all the models
    models=[]
    #Results of Naive Bayes
    models.append(NaiveBayes_Opt())
    #Results of Bagging
    models.append(Bagging_Opt())
    #Results of KNN
    models.append(KNN_Opt())
    #Results of Random Forest
    models.append(RandomForest_Opt())
    #Results of Gradient Boost
    models.append(GradientBoost_Opt())
    #Results of SVM
    models.append(SVM_Opt())
    #Results of Decision Trees
    models.append(DecisionTrees_Opt())
    #Results of AdaBoost
    models.append(AdaBoost_Opt())
    #Results of XGBoost
    models.append(XGBoost_Opt())
    #Sort the array
    models.sort()
    #Arrange in Descending Order
    models.reverse()
    print("Report:")
    print("Model : Accuracy")
    for i in models:
        print(i[1],' : ',i[0]*100)
    print('Selected Model: ',models[0][1])
    print("Model Accuracy:", models[0][0])

model_selection()