
# coding: utf-8

# In[22]:


#Loading the Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
get_ipython().magic('matplotlib inline')


# In[23]:


customer_data = pd.read_csv("CustomerUsage.csv")


# In[3]:


customer_data.head()


# In[24]:


## Data Cleaning and filtering 
#dropping an Column not needed 
print('number of columns  before :{}'.format(len(customer_data.columns)))
customer_data.drop('Unnamed: 0', axis=1, inplace=True)
print('number of columns  After :{}'.format(len(customer_data.columns)))


# In[25]:


customer_data.head()


# In[26]:


#df_output['onboarded date'] = pd.to_datetime(df_output['onboarded date'])
customer_data['signup_date'] = pd.to_datetime(customer_data['signup_date'])
customer_data.head()


# In[27]:


customer_data['signup_date']= pd.to_datetime(customer_data['signup_date']).dt.date
customer_data["Age"] = (dt.datetime.now().date()- customer_data['signup_date'])
customer_data['Age'] = customer_data.apply(lambda row: row.Age.days,axis=1)


# In[8]:


customer_data.head()


# In[28]:


customer_data.columns


# In[29]:


customer_data.info()


# In[30]:


customer_data.describe()


# In[31]:


customer_data['converted_to_paid'] = np.where(customer_data['type']== "Paid",1,0)


# In[32]:


customer_data.head()


# In[33]:


#Now lets see how ord or new subscrition age we have  
customer_data["Age"].plot.hist(bins=50)


# In[34]:


###
sns.jointplot(x="Age" ,y='logins',data =customer_data)


# In[ ]:


#Jointplot Age and urlcount by unique customers 
# Add 2 columns  
DailyTimeSpent  -  1-150
URLHitCount  -  100 - 2549

sns.jointplot(x="Age",y="DailyTimeSpent " , data =customer_data,kind = 'kde')


# In[ ]:


#Jointplot Age and urlcount by unique customers 
# Add 2 columns  
DailyTimeSpent  -  1-150
URLHitCount  -  100 - 2549

sns.jointplot(x="DailyTimeSpent",y="URLHitCount " , data =customer_data,kind = 'kde')


# In[35]:


#Regression model  

from sklearn.cross_validation import train_test_split


# In[36]:


customer_data.head()


# In[37]:


customer_data.columns


# In[38]:



X = customer_data[['users','logins','pipelines', 'templates', 'deployments']]
y= customer_data['converted_to_paid']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[43]:


predictions = model.predict(X_test)


# In[44]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[47]:



df2 =pd.DataFrame({"Prediction": predictions, "Actual": y_test})


# In[49]:


df2[df2["Prediction"]!= df2["Actual"]]

