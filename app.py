
# coding: utf-8


#Loading the Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from flask import Flask, request, redirect, url_for, jsonify

app = Flask(__name__)

customer_data = pd.read_csv("CustomerUsage.csv")

def dataPrep(): 
    
    print(customer_data.head())
    ## Data Cleaning and filtering 
    #dropping an Column not needed 
    print('number of columns  before :{}'.format(len(customer_data.columns)))
    customer_data.drop('Unnamed: 0', axis=1, inplace=True)
    print('number of columns  After :{}'.format(len(customer_data.columns)))
    #df_output['onboarded date'] = pd.to_datetime(df_output['onboarded date'])
    customer_data['signup_date'] = pd.to_datetime(customer_data['signup_date'])
    customer_data.head()
    customer_data['signup_date']= pd.to_datetime(customer_data['signup_date']).dt.date
    customer_data["Age"] = (dt.datetime.now().date()- customer_data['signup_date'])
    customer_data['Age'] = customer_data.apply(lambda row: row.Age.days,axis=1)
    customer_data['id']= customer_data.index
    customer_data['customer_name2'] = customer_data.apply(lambda row: "company"+ str(row.id) , axis=1)
    print(customer_data.columns)
    print(customer_data.info())
    print(customer_data.describe())
    customer_data['converted_to_paid'] = np.where(customer_data['type']== "Paid",1,0)


dataPrep()

print(customer_data.head())

df2 = pd.DataFrame()
@app.route('/', methods=['GET', 'POST'])
def predictProspects():
    X = customer_data[['users','logins','pipelines', 'templates', 'deployments']]
    y= customer_data['converted_to_paid']

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42)


    model = LogisticRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    from sklearn.metrics import classification_report,confusion_matrix
    #print(classification_report(y_test,predictions))
   # print(confusion_matrix(y_test,predictions))


    df2 =pd.DataFrame({"Prediction": predictions, "Actual": y_test})

    print(df2[df2["Prediction"]!= df2["Actual"]])
    data = df2.to_dict()
    

    return jsonify(data)


@app.route("/paidCustomers")
def paidCustomers():
    paid_customers = list(customer_data[customer_data['type']=='Paid']['customer_name'])
    return jsonify(paid_customers)


@app.route("/allCustomers")
def allCustomers():
    all_customers = list(customer_data['customer_name'])
    return jsonify(all_customers) 


@app.route("/predictedCustomers")
def predictedCustomers():
    
    predicted_customers = list(df2[df2["Prediction"]!= df2["Actual"]])
    return jsonify(predicted_customers)   

if __name__ == "__main__":
    app.run(debug=True)
