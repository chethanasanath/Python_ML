#This is part1 of LogisticRegression. Here we clean the data, fill in missing age columns, drop cabin column
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("titanic_train.csv") #This training data is from Kaggle.com
print(train.head())

print(train.isnull())#This is going to print all boolean values, i.e., true if the particular field value in the csv
#is null and false if not null

#With above kind of true/false values we can create a heatmap
#sns.heatmap(train.isnull(),cmap='viridis')
#plt.show()#Every yelow mark seen in the plot indicates true value. Yes, it is true and it is null
#From the plot it is evident that,there is some age data that is missing and we could probably make it up with the
#help of other data that is available. There is a lot of Cabin data that is missing and we may not be able to make up to it.
#This is called cleaning of data, lets do it below

sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Sex',data=train)
#plt.show()

#Lets get an idea of the age group of the people who Survived in the titanic dataset
sns.distplot(train['Age'].dropna(),bins=30)

#The above is through seaborn, we could get the same through pandas below
#train['Age'].hist(bins=50)
#plt.show()

#sns.boxplot(x='Pclass',y='Age',data=train)
#plt.show()#This boxplot shows the avg age of Passengers across Pclass (passenger class). The line bisecting the box reps avg age

#At line14 above when we plotted heatmap for train.isnull, we saw that there is some age data that is missing. Let us try to
#make up for this by calculating avg age using pandas. Lets write below fn for that, this is based on analyzing boxplot result in line 32

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

#Lets apply the above fn
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#Now lets checkout the heatmap
sns.heatmap(train.isnull(),cmap='viridis')
#plt.show()
print("Printing Age column1")
print(train['Age'])

#Further Cabin column has lot of null values, lets drop this
train.drop(['Cabin'],inplace=True,axis=1)
#plt.show()






#Below is part2 of LogisticRegression, where we further clean the data. Here, we convert columns in training data that have continuous values into
# categorical values and also drop out unused columns from the pd.
#Below we are creating dummy variables for required fields. For ex, 'Sex'. This field has the categorical value of
#Male or Female. A ML algorithm would not be able to interpret Male or Female which is a string. We have to create a new column indicating
#0 or 1 and this is called creating dummy variable and this could be understood by ML
#We'll do a similar thing to 'Embarked' column
sex = pd.get_dummies(data=train['Sex'])
print(sex)#This output shows that when it is male there is 1 for it and similarly for female. There is one issue with this feature, i.e.,
#each column here (male or female) is a perfect detector of the other known as Multicollinearity. We would have issue with this. This will mess up
# the algoritm because a bunch of columns would be a perfect predictors of other column. To avoid this we pass the second arg in above line (drop_first)

sex = pd.get_dummies((train['Sex']),drop_first=True)#This would drop the first column and this is exactly what we want and would feed this in
#as a column to ML algorithm


embarked = pd.get_dummies(train['Embarked'],drop_first=True)
#Now that we have encoded the columns sex and embarked, lets add them to our dataframe by using concat
train = pd.concat([train,sex,embarked],axis=1)#axis=1 means applicable across all columns
#Now we could drop the columns that we dont use
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'],axis=1,inplace=True)
print(train)
#Now this data looks perfect for a ML algorithm

#We could further delete PassengerId column as it is essentially just the index column which we already have
train.drop(['PassengerId'],axis=1,inplace=True)
print(train.head())





##Below is part3 of LogisticRegression. Here we train the model and use it for prediction
y=train['Survived']
x=train.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
logModel = LogisticRegression()
logModel.fit(X_train, y_train)
predictions = logModel.predict(X_test)

#Now lets evaluate the model
from sklearn.metrics import classification_report
#We have seen that confusion matrix is used for evaluating a LogisticRegression Model. Then, what is this classification_report? This helps,
#in providing us the accuracy, precision, which we calculate ourselves in case of a confustion matrix
print(classification_report(y_test,predictions))#pass in the true y value and the predictions here
#Here we are with below values of Precision and recall.
#We could further explore by doing the same with the train and test data CSVs and then compare the 2 results for precision and recall
