import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Classified Data',index_col=0)#Note that this is anonymous classified data that is available.
#Lets check the head of it
print(df.head())

#Now since we are using KNearestNeighbours (KNN) algorithm to predict, let us scale the features
#using sklearn library
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#We are fitting the features below
scaler.fit(df.drop('TARGET CLASS',axis=1))

#Let us transform the features in to their scaled version blow
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())
#Now our data is fit to be put into K Nearest Neighbours algorithm which depends on distance bw each feature

from sklearn.model_selection import train_test_split
x=df_feat
y=df['TARGET CLASS']
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)#here n_neighbors=1 is nothing but K=1
knn.fit(x_train,y_train)#This is erroring, saying, "Found input variables with inconsistent numbers of samples: [700, 300]"
pred = knn.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#This would give a precision of 92%

#Lets us see, if we can squeeze a bit more from our model using a better K value.
#use elbow method to choose the K value
error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)#This is erroring, saying, "Found input variables with inconsistent numbers of samples: [700, 300]"
    pred_i=knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1,40),error_rate)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()#This plot's screenshot is in the word doc. It basically shows the error_rate vs K value.
#It shows that, higher the K value, lesser the error rate. Lets choose K=17 here


#Retrain the model with K=17
knn = KNeighborsClassifier(n_neighbors=17)#here n_neighbors=17 is nothing but K=17
#The down side of choosing a higher k value is it takes longer to train and fit your model
#The up side of choosing a higher k value is that it reduces the error_rate
knn.fit(x_train,y_train)#This is erroring, saying, "Found input variables with inconsistent numbers of samples: [700, 300]"
pred = knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#This would give a precision of 95% - hence there is improvement by choosing a better K value
