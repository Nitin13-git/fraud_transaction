

# # Import all required libraries

# In[2]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
#from plotly.offline import iplot
#import plotly as py
#import plotly.tools as tls
#import cufflinks as cf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#('pip install cassandra-driver')


# In[4]:


from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
        'secure_connect_bundle': r'C:\Users\welcome\Desktop\secure-connect-fraud-detection-project.zip'
}
auth_provider = PlainTextAuthProvider('sucxMwwdJsamJapdFOdcUsgy', 'KB9lDaTEf_v,12OInbgmPo6zr+b1hOdyOoorlbjNbM3ZD0C_TZqIdAmZ.a8Lq+6RWlk83Fj_,nMvNZX4s7ZdCZx_BCZ68oLoOeZMOH,vl_ZJa24GUB0d9-3ng8tEmXA5')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace("ML_Project")
session.default_fetch_size=None

count=0
results = session.execute("SELECT * FROM creditcard",timeout=None)
for i in results:
    count+=1
    data=pd.DataFrame(results)


# In[5]:


data


# In[117]:


X = data.drop('Class', axis = 1)
Y=data["Class"]


# In[118]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# In[119]:


x_train


# In[120]:


x_test


# In[121]:


y_train


# In[122]:


y_test


# In[123]:


d_train = pd.concat([x_train,y_train], axis =1)
d_train


# In[124]:


class0 = d_train[d_train['Class']==0]

class1 = d_train[d_train['Class']==1]


# In[125]:


class0.head()


# In[126]:


class1.head()


# In[127]:


frames = ['Time', 'Amount']
x= d_train[frames]
y=d_train.drop(frames, axis=1)


# In[128]:


x.head()


# In[129]:


y.head()


# In[130]:

scaler = StandardScaler()
temp_col=scaler.fit_transform(x)
pd.DataFrame(temp_col)


# In[131]:


scaled_col = pd.DataFrame(temp_col, columns=frames)
scaled_col


# In[132]:


d_temp = d_train.drop(frames, axis=1)
d_temp


# In[133]:


d_temp.reset_index()


# In[137]:


d_=d_temp.reset_index().drop("index",axis=1)
d_.head()


# In[153]:


d_scaled = pd.concat([scaled_col, d_], axis =1)
d_scaled


# In[162]:


X___= d_scaled.drop('Class', axis = 1)
Y___=pd.DataFrame(d_scaled["Class"])
Y___


# In[177]:


"""# Dimensionality Reduction"""

from sklearn.decomposition import PCA

pca = PCA(n_components=15)

X_temp_reduced = pca.fit_transform(d_scaled)


# In[178]:


X_temp_reduced = pca.fit_transform(X___)
X_reduce=pd.DataFrame(X_temp_reduced)
X_reduce


# In[157]:


pca.explained_variance_ratio_


# In[158]:


pca.explained_variance_


# In[165]:


new_data=pd.concat([X_reduce,Y___],axis=1)
new_data


# In[166]:


new_data.to_csv('final_data.csv')


# In[199]:


X_train, X_test, y_train, y_test= train_test_split(X___, Y___['Class'], test_size = 0.25, random_state = 42)


# In[200]:


print(X_train.shape)
print(X_test.shape)


# In[202]:


print(y_train.shape)
print(y_test.shape)


# In[203]:


sns.countplot('Class', data=d_scaled)


# In[204]:


#'pip install --user imblearn'


# In[205]:


print(y_train.value_counts())
print(y_test.value_counts())


# In[206]:


from collections import Counter 
Counter(y_train)


# ### SMOTE (Synthetic Minority Oversampling Technique) – Oversampling

# In[247]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(0.5,random_state = 42)


# In[248]:


X_train_Smote, Y_train_Smote=sm.fit_resample(X_train,y_train)


# In[249]:


X_train_Smote


# In[250]:


d_smote = pd.concat([X_train_Smote,Y_train_Smote], axis =1)
d_smote


# In[251]:


print("The Number of classes Before the fit {}".format(Counter(y_train)))
print("The Number of classes After the fit {}".format(Counter(Y_train_Smote)))


# In[252]:


sns.countplot('Class', data=d_smote)


# ### RANDOM FOREST

# In[253]:


from sklearn.ensemble import RandomForestClassifier
rf_SMOTE = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=20,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rf_SMOTE


# In[254]:


rf_SMOTE.fit(X_train_Smote, Y_train_Smote)


# In[255]:


y_pred_regressor_rf1 = rf_SMOTE.predict(X_test)
y_pred_regressor_rf1


# In[256]:


RF_Accuracy_SMOTE = rf_SMOTE.score(X_test,y_test)
RF_Accuracy_SMOTE


# In[257]:


cm_regressor_rf1 = confusion_matrix(y_test, y_pred_regressor_rf1)
print(cm_regressor_rf1)
print("Accuracy score of the model:",accuracy_score(y_test, y_pred_regressor_rf1))
print(classification_report(y_test, y_pred_regressor_rf1))


# In[271]:


import pickle


# In[274]:


file = open('Credit_Fraud_Detection_.pkl', 'wb')
pickle.dump(rf_SMOTE, file)


# In[275]:


f = pd.read_pickle(r'Credit_Fraud_Detection_.pkl')
f





