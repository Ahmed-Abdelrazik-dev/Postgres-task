import pandas as pd
import numpy as np
import pickle
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://iti:iti@localhost:5432/data_management')

con = engine.connect()
table_name = 'iris'

table_df = pd.read_sql(table_name , con = engine , columns = ['sepal_length'
                                                        ,'sepal_width'
                                                        ,'petal_length'
                                                        ,'petal_width'
                                                        ,'species'])

x = table_df.drop(['species'],axis=1)

y = table_df['species']

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# save the model to disk
filepath = '/home/iti/Desktop/Task5/ML_model.sav'
pickle.dump(knn, open(filepath, 'wb'))
