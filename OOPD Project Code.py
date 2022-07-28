#!/usr/bin/env python
# coding: utf-8

# # OOPD PROJECT : CB -  ACP Design and Prediction.

# In[ ]:


# Group Members:- Aman Srivastava (MT202333)
#                 Akanksha Jarwal (MT20331)
#                 Amaithi Priya (MT20332)
#                 Himanshi Garg (MT20340)


# In[1]:


# import libraries
from tensorflow import keras
import sklearn
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score


# In[2]:


# Profiling Code Starts
import cProfile,pstats
profiler = cProfile.Profile()
profiler.enable()


# # Database Connection

# In[3]:


class DB_Conn:
    def __init__(self):
        self.connection = None                              # Encapsulation
        self.__host_name = "localhost"    
        self.__username = "root"
        self.__pw="Root@123"  # SQL Terminal Password ####  
        self.db="ACP_detection"  # Database Name #####
    
    def set_credentials(self, username, password, database):
        self.username = username
        self.pw = password
        self.db = database
    
    # This function creates connection with the server.
    def create_server_connection(self):
        try:
            connection= mysql.connector.connect(
                host=self.__host_name,
                user=self.__username,
                password=self.__pw
            )
            print("Server connection successful")
        except Error as err:
            print(f"Error: '{err}'")
        return connection
    
    # This function creates connection with the database.
    def create_db_connection(self):
        connection= None
        try:
            connection= mysql.connector.connect(
                host=self.__host_name,
                user=self.__username,
                password=self.__pw,
                database= self.db)
            print("Database connection successful")
        except Error as err:
            print(f"Error: '{err}'" )
        return connection


# # Database Creation using Query

# In[4]:


class DB_Manager(DB_Conn):    # Inheritence
    def __init__(self):
        self.db_create_query= "Create database ACP_detection" ### Database creation Query
    
    def get_db_name(self, db_name):
        self.db_create_query = "Create database " + db_name
    
    # This Function creates a database.
    def create_database(self, connection):
        cursor= connection.cursor()
        try:
            cursor.execute(self.db_create_query)
            print("Database created successfully")
        except Error as err:
            print(f"Error: '{err}'" )
    
    #This function executes the query
    def execute_query(self,connection,query):
        cursor= connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            print("Query was successful")
        except Error as err:
            print(f"Error: '{err}'" )
        return result
    
    # This function loads the data in the database.
    def load_data(self, connection, data):
        cur= connection.cursor()
        s= "INSERT INTO samples (Peptide_sequence, label) VALUES (%s,%s)"
        cur.executemany(s, data)
        connection.commit()
        print("Data Loaded Successfully")


# # Accessing Database using Objects

# In[5]:


DBC = DB_Conn()


# In[6]:


server_connection = DBC.create_server_connection()


# In[7]:


DBM = DB_Manager()
DBM.create_database(server_connection)


# In[8]:


db_connection = DBC.create_db_connection()


# In[9]:


# Creating Table in the database to store Sample records.
create_table= """
create table samples(
Peptide_sequence varchar(200),
label int);
"""
#connect to the database
DBM.execute_query(db_connection,create_table)


# # Reading Fasta files

# In[62]:


# Reading the input Fasta File

#Negative samples
with open(r'C:\Users\mailt\Desktop\fasta\fasta\data\negative\balanced.fasta','r') as a:
    x = a.read()
    
    
#Positive Samples
with open(r'C:\Users\mailt\Desktop\fasta\fasta\data\positive\balanced.fasta','r') as e:
    u= e.read()
    


# # Data Pre processing 

# In[12]:


# Data Pre processing 

class Data_Preprocessor:
    def __init__(self):
        pass
    
    def preprocess(self, data, cls): 
        s= data.split("\n")
        n= len(s)

        arr=[]
        while n>0:
            arr.append(s[n-2])
            n= n-2
        
        lst=[]
        for i in range(len(arr)-1):
            lst1=[]
            a=()   
            lst1.append(arr[i])
            lst1.append(cls)
            a= tuple(lst1)
            lst.append(a)

        return lst


# In[13]:


preprocessor = Data_Preprocessor()
lst = preprocessor.preprocess(x, 0)
lst1 = preprocessor.preprocess(u, 1)


# # Loading Samples in SQL Table

# In[14]:


DBM.load_data(db_connection, lst)         ## Negative Samples
DBM.load_data(db_connection, lst1)        ##Positive Samples


# In[15]:


# calling all the samples from database.
q1="""
select * from samples;
"""
results=DBM.execute_query(db_connection, q1)


# In[16]:


# Converting the data to a dataframe.

#create data frame 

df=[]

for res in results:
    result= list(res)
    df.append(res)
    
columns= ["Peptide_sequence", "label"]
df= pd.DataFrame(df,columns=columns )

display(df)


# In[17]:


# Loading the p-feature analysis file

d2=pd.read_csv(r"C:\Users\mailt\Desktop\OOPD project\only_dipeptid (1).csv")
d2.head()
pf = d2.reindex(np.random.permutation(d2.index))


# In[18]:


#Preprocessing the file using Standard Scaler
sc=StandardScaler()
c2=pd.DataFrame(sc.fit_transform(pf))
c2.columns=pf.columns
c2.head()


# In[19]:


Y1=c2.Label


# # Feature Selection using Variance Threshold

# In[20]:


vt=VarianceThreshold(1.0)
fs=vt.fit_transform(c2)
fs=c2.columns[vt.get_support(indices=True)]
c3=c2[fs]
c3.head()


# In[21]:


# Removing NA values and calculating the mean of dataframe and replacing NA with mean.
m=round(c3.mean(),6)
m
c3.fillna(m,inplace=True)
c3.head(5)


# In[22]:


# Splitting the Dataset usinig train_test_split with training size = 70% and Testing size= 30%
X_train, X_test, Y_train, Y_test = train_test_split(c3, Y1, test_size=0.30, random_state= 22) 


# # Decision Tree Classifier

# In[23]:


class DecisionTreeClassifierModel:
    def __init__(self):
        self.Model=None
        self.prediction_dt= None
        self.predictt=None
        self.accuracy= 0
        self.f1score= 0
        
    # This function trains the model on Decision Tree Classifier.
    def fit(self,X_train, Y_train):
        self.Model= DecisionTreeClassifier(random_state=15)
        self.Model= self.Model.fit(X_train,Y_train)
        
        return self.Model
    
    # This function predicts the values in 0 and 1.
    def predict_values(self, X_test):
        self.prediction_dt= self.Model
        self.prediction_dt = self.prediction_dt.predict(X_test)
        arr=[]
        for i in self.prediction_dt:
            if i <0:
                arr.append(0)
            else:
                arr.append(1)
        
        return arr
    
    # This function returns the prediction values of X_test
    def predict(self,X_test):
        self.predictt= self.Model.predict(X_test)        
        
        return self.predictt 
    
        


# In[24]:


DTC= DecisionTreeClassifierModel()
aa=DTC.fit(X_train, Y_train)
aa


# In[25]:


aa1= DTC.predict_values(X_test)
aa1


# In[26]:


predict= DTC.predict(X_test)
acc1= (accuracy_score(predict,Y_test))
print("Accuracy of Decision Tree: ",acc1)
fs1= (f1_score(predict,Y_test))
print("f1_score of Decision Tree: ",fs1)


# # Random Forest Classifier

# In[27]:


class RandomForestModel:
    def __init__(self):
        self.Model=None
        self.prediction_dt= None
        self.predictt=None
        self.accuracy= 0
        self.f1score= 0
        
        
    # This function trains the model on Random Forest Classifier.
    def fit(self,X_train, Y_train):
        self.Model= RandomForestClassifier(random_state=18) #max_depth=2,max_features='auto',n_estimators=3,
        self.Model= self.Model.fit(X_train,Y_train)
        
        return self.Model
    
    #This function checks for a random value and generates the score.
    def Random(self,X_train, Y_train):
        
        return self.Model.score(X_train,Y_train)
    
     # This function predicts the values in 0 and 1.
    def predict_values(self, X_test):
        self.prediction_dt= self.Model
        self.prediction_dt = self.prediction_dt.predict(X_test)
        arr=[]
        for i in self.prediction_dt:
            if i <0:
                arr.append(0)
            else:
                arr.append(1)
        
        return arr
    
    # This function returns the values in form of Probability.
    def probability(self,X_test):
        return self.Model.predict_proba(X_test)[:,1]
    
    # This function returns the prediction values of X_test
    def predict(self,X_test):
        self.predictt= self.Model.predict(X_test)        
        
        return self.predictt 
    


# In[28]:


RFM=RandomForestModel()


# In[29]:


RF=RFM.fit(X_train, Y_train)
RF


# In[30]:


RFM.Random(X_train,Y_train)


# In[31]:


# Predicting value in form of 0 and 1.
RFMpvalue=RFM.predict_values(X_test)


# In[32]:


#Predicting the Probability Values
RFMprob=RFM.probability(X_test)


# In[33]:


# Calculating the accuracy score of our Model.
predict1= RFM.predict(X_test)
accRF= (accuracy_score(predict1,Y_test))
print("Accuracy of Random Forest: ",accRF)

# Calculating the f1 Score of Our Random Forest Model.
f1RF= (f1_score(predict,Y_test))
print("f1_score of Random Forest: ",f1RF)


# In[34]:


# Plotting a Confusion Matrix of Our result.

print('Confusion matrix')
print(confusion_matrix(Y_test, RF.predict(X_test)))
    #confusion matrix
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(Y_test, RF.predict(X_test)), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[35]:


# calculating the fpr and tpr values.
y_pred_keras = RFMprob.ravel()
y_test_kears= Y_test.ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_kears, y_pred_keras)


# In[36]:


# Calculating the auc score of our RF Model.
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)


# In[37]:


# Plotting the ROC curve

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[38]:


class Conf_metrix:
    # Creating a function to report confusion metrics
    def confusion_metrics (self,conf_matrix):
    # save confusion matrix and slice into four pieces
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)
    
    # calculate accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
        conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
        conf_specificity = (TN / float(TN + FP))
    
        # calculate precision
        conf_precision = (TN / float(TN + FP))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        print('-'*50)
        print(f'Accuracy: {round(conf_accuracy,2)}') 
        print(f'Mis-Classification: {round(conf_misclassification,2)}') 
        print(f'Sensitivity: {round(conf_sensitivity,2)}') 
        print(f'Specificity: {round(conf_specificity,2)}') 
        print(f'Precision: {round(conf_precision,2)}')
        print(f'f_1 Score: {round(conf_f1,2)}')


# In[39]:


mat= Conf_metrix()
res=confusion_matrix(Y_test, RF.predict(X_test))
mat.confusion_metrics(res)


# # Saving the values to the database

# In[40]:


# created a Table in the database named results.
create_table= """
create table Results(
Peptide_sequence varchar(200),
label int);
"""
#connect to the database
DBM.execute_query(db_connection,create_table)


# In[41]:


# processing the final Data before loading in the database.
preprocessor = Data_Preprocessor()
aY = [1 if i == 1 else 0 for i in RF.predict(X_test)]
testindex=X_test.index

lst3=[]
for i in range(len(aY)):
    lst2=[]
    a=()
    b= testindex[i]
    lst2.append(df.Peptide_sequence[b])
    lst2.append(aY[i])
    a= tuple(lst2)
    lst3.append(a)


# ### 

# In[42]:


class DB_Load:
    def __init__(self):
        pass
    
    def load_data1(self, connection, data):
        cur= connection.cursor()
        s= "INSERT INTO results (Peptide_sequence, label) VALUES (%s,%s)"
        cur.executemany(s, data)
        connection.commit()
        print("Data Loaded Successfully")


# In[43]:


# Loading the final processed data in the database.
load= DB_Load()
load.load_data1(db_connection,lst3)


# In[45]:


# Profiling Code end
import io
profiler.disable()
s = io.StringIO()
stats = pstats.Stats(profiler,stream=s).sort_stats('ncalls')
stats.print_stats()
with open('test.txt', 'w+') as f:
    f.write(s.getvalue())


# # End

# In[ ]:




