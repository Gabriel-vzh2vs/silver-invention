# Import List
import pandas as pd
import streamlit as st
import numpy as np 
import plotly
import plotly.express as px 
import pymysql
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_label_graph import label_graph, LabelConfig
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from math import sqrt
############################################################# (Sklearn Set) ######################################################################################
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
################################################################ (Gabe Core Dataframes) ###################################################################
st.set_page_config(layout="wide")
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="material",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )
    return selection

# Dataframe: Revenue

connection_string = "mysql+pymysql://username:password@IP/Database" #Replace with Server-associated data.
engine = create_engine(connection_string)
query = 'SELECT * FROM payments'
df = pd.read_sql_query(sql=text(query), con=engine.connect())
revenue = round(sum(df.amount),2)

query = 'SELECT * FROM orderdetails'
orderdetails = pd.read_sql_query(sql=text(query), con=engine.connect())
expenses = 0
i = 0
for i in orderdetails:
    orderdetails['product'] = (orderdetails['quantityOrdered'] * orderdetails['priceEach'])
expenses = sum (orderdetails['product'])
salary_ranges = pd.cut(orderdetails['product'], bins=range(0, 15000, 1000), include_lowest=True)
grouped_data = orderdetails.groupby(salary_ranges)['product'].mean()

# Dataframe: dfq

engine = create_engine(connection_string)
query = 'SELECT * FROM orders INNER JOIN orderdetails ON orders.orderNumber = orderdetails.orderNumber;' # Inner Join
dfz = pd.read_sql_query(sql=text(query), con=engine.connect()) 
expenses = 0 
i = 0
for i in dfz:
    dfz['expenses'] =  (dfz['quantityOrdered'] * dfz['priceEach'])
expenses= round(sum(dfz['expenses']),2)
dfz = dfz.T.drop_duplicates().T
dfz1 = pd.merge(dfz, df, left_on='customerNumber', right_on = 'customerNumber', suffixes=('_x', '_y')) # Merge Dataframes!
dfz1 = dfz1.T.drop_duplicates().T # Remove duplicates
dfq = dfz1.groupby(["orderNumber"], as_index= 'TRUE').sum().reset_index()
dfq1 = dfz1.groupby(["paymentDate"], as_index= 'TRUE').sum().reset_index()

# Dataframe dfconnection2
query = "SELECT * FROM customers INNER JOIN orders ON customers.customerNumber = orders.customerNumber"
dfconnection2 = pd.read_sql_query(sql=text(query), con=engine.connect())

# Dataframe: dfconnection3

query = "SELECT * FROM orders INNER JOIN orderdetails ON orderdetails.orderNumber = orders.orderNumber"
dfconnection3 = pd.read_sql_query(sql=text(query), con=engine.connect())
dfconnection3 = dfconnection3.T.drop_duplicates().T
dfconnection3['sales'] = dfconnection3['quantityOrdered'] * dfconnection3['priceEach']
dfconnection3m = pd.merge(dfconnection3, dfconnection2, left_on='orderNumber', right_on = 'orderNumber', suffixes=('_x', '_y')) # Merge Dataframes!
dfconnection3m = dfconnection3m.T.drop_duplicates().T # Remove duplicates
new_df = dfconnection3[['orderNumber', 'sales']]
new_df = new_df.T.drop_duplicates().T
grouped_df = new_df.groupby("orderNumber", as_index = 'TRUE').mean().reset_index()

##################################################################### (Olivia Code Restricted Area) ############################################

# Dataframe: Employees
query = 'SELECT * FROM employees'
employees = pd.read_sql_query(sql=text(query), con=engine.connect())

# Dataframe: Payments
query = 'SELECT * FROM payments'
payments = pd.read_sql_query(sql=text(query), con=engine.connect())
revenue = sum(payments.amount)

# Dataframe: orderDetails
query = 'SELECT * FROM orderdetails'
orderDetails = pd.read_sql_query(sql=text(query), con=engine.connect())
expenses = 0 
i = 0
for i in orderDetails:
    orderDetails['expenses'] = (orderDetails['quantityOrdered'] * orderDetails['priceEach'])
    orderDetails['salary'] = round((orderDetails['quantityOrdered'] * orderDetails['priceEach'] * 0.20),2)
    dfconnection3m['salary'] = round((orderDetails['quantityOrdered'] * orderDetails['priceEach'] * 0.20),2)
expenses = sum(orderDetails['expenses'])
profit_margin = ((revenue - expenses) / expenses)

########################################################################(Extras)###########################################################################

# Setup for Cov() Matrix:
cols = ['quantityOrdered', 'priceEach', 'expenses', 'amount']
X = dfq[cols]
X_scaled = (X - X.mean(axis=0))/X.std(axis=0) # Scale and Normalize the values.
cov_matrixQPEA = X_scaled.cov()
sns.set(font_scale=0.8)
sns.heatmap(cov_matrixQPEA, annot=True)

cols = ['orderNumber', 'customerNumber', 'expenses', 'amount']
X = dfq[cols]
X_scaled = (X - X.mean(axis=0))/X.std(axis=0) # Scale and Normalize the values.
cov_matrixOCEA = X_scaled.cov()
sns.heatmap(cov_matrixOCEA, annot=True)

##########################################################################################################################################################

# Setup for plotting The History Graph

config: LabelConfig = {
    'categories': [
        {'key': 'Negative Trend', 'color': 'rgba(255, 110, 110, 0.1)'},
        {'key': 'Positive Trend', 'color': 'rgba(110, 110, 255, 0.1)'}
    ]
}

##########################################################################################################################################################
#Streamlit Setup:

st.header("Dashboard")
page = st.sidebar.selectbox('Select page',
  ['Revenue per Transaction','Salary Optimization', 'Expense-Revenue Optimization', 'Profit Analysis'])

if page == 'Revenue per Transaction':
  ## Revenue per Transaction
  clist = ['Distributions', 'Grouped Sales', 'Interactive Table']
  selection = st.selectbox("Select a Subject:",clist)
  if selection == 'Distributions':
    col1, col2 = st.columns(2)
    fig = px.histogram(grouped_df, x="sales", title ='Distribution of Sales')
    
    col1.plotly_chart(fig,use_container_width = True)
    fig = px.histogram(dfq, x="quantityOrdered", title ='Distribution of Product Orders')
    
    col2.plotly_chart(fig,use_container_width = True)

  if selection == 'Interactive Table':
     selectionb = aggrid_interactive_table(df=dfq1)

  if selection == 'Grouped Sales':
    col1, col2 = st.columns(2)
    fig = px.bar(grouped_df, x='orderNumber', y='sales', title='Sales per Account')
    
    col1.plotly_chart(fig,use_container_width = True)
    fig = px.bar(dfq, x='customerNumber', y='amount', title='Sales per Customer')
    
    col2.plotly_chart(fig,use_container_width = True)


if page == 'Salary Optimization':
  ## Salary Optimization
  clist = ['Regressions Prediction', 'Salary Information', 'Secondary Salary Information']
  selection = st.selectbox("Select a Group:", clist)

  if selection == "Regressions Prediction":
    w1 = st.sidebar.checkbox("Show table", False) #Show Table
    scatterplot = st.sidebar.checkbox("Show Scatterplot", False)
    plothist = st.sidebar.checkbox("Show Histograms", False)
    trainmodel = st.sidebar.checkbox("Train Model", False)
    dokfold = st.sidebar.checkbox("DO KFold", False)
    linechart = st.sidebar.checkbox("Line Charts",False)

    if w1: # Garbage, but Works.
        st.dataframe(dfconnection3m,width=2000,height=500)

    if linechart: 
        col1, col2 = st.columns(2)
        options = ("salary","quantityOrdered","priceEach")
        sel_cols = st.selectbox("Select columns", options, 1)
        st.subheader("Line chart")
        fig = st.line_chart(data = dfconnection3m, x = sel_cols, y = 'sales', height = 600)

    if plothist: # This works!
        st.subheader("Distributions of each Category")
        options = ("quantityOrdered", "priceEach", "sales", 'salary')
        sel_colsx = st.selectbox("Select Category", options,1)
        st.write(sel_colsx)
        fig = go.Histogram(x=dfconnection3m[sel_colsx],nbinsx=50)
        st.plotly_chart([fig])

    if scatterplot:
        st.subheader("Correlation between Selected Variable and Sales")
        options = ("orderNumber","quantityOrdered","priceEach","sales")
        w7 = st.selectbox("Variable", options)
        st.write(w7)
        px.scatter(data_frame = dfconnection3m, x = dfconnection3m[f'{w7}'], y = dfconnection3m["sales"])

    if trainmodel:
        st.header("Modelling")
        options = ("quantityOrdered","priceEach", "sales", "customerNumber_x", "creditLimit", "salary", "salesRepEmployeeNumber")
        sel_colsx = st.selectbox("Select columns", options, 1)
        y=dfconnection3m.sales
        X=dfconnection3m[[f'{sel_colsx}']].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        lrgr = LinearRegression()
        lrgr.fit(X_train,y_train)
        pred = lrgr.predict(X_test)

        mse = mean_squared_error(y_test,pred)
        rmse = sqrt(mse)

        st.markdown(f"""
        Linear Regression model trained :
            - MSE:{mse}
            - RMSE:{rmse}
        """)
        st.success('Model trained successfully')

    if dokfold:
        st.subheader("KFOLD Random sampling Evalution")
        my_bar = st.progress(0)
        X=dfconnection3m.values[:,-1].reshape(-1,1)
        y=dfconnection3m.values[:,-1]
        #st.progress()
        kf=KFold(n_splits=10)
        mse_list=[]
        rmse_list=[]
        r2_list=[]
        idx=1
        fig=plt.figure()
        i=0
        for train_index, test_index in kf.split(X):
        #	st.progress()
            my_bar.progress(idx*10)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lrgr = LinearRegression()
            lrgr.fit(X_train,y_train)
            pred = lrgr.predict(X_test)
            
            mse = mean_squared_error(y_test,pred)
            rmse = sqrt(mse)
            r2=r2_score(y_test,pred)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            plt.plot(pred,label=f"dataset-{idx}")
            idx+=1
        plt.legend()
        plt.xlabel("Data points")
        plt.ylabel("PRedictions")
        plt.show()
        st.plotly_chart(fig)

        res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
        res["MSE"]=mse_list
        res["RMSE"]=rmse_list
        res["r2_SCORE"]=r2_list

        st.write(res)
  
  if selection == 'Salary Information':
       col1,col2 = st.columns(2)
       fig = px.histogram(orderDetails, x= "salary", title = 'Distribution of Commission', labels = {'salary': 'Commissions'})
       col1.plotly_chart(fig)
       fig = px.bar(dfconnection3m, x= "salesRepEmployeeNumber", y = "salary", title ='Commissions per Employee')
       col2.plotly_chart(fig, use_container_width = True)

  if selection == 'Secondary Salary Information':
       col1, cols2 = st.columns(2)
       fig = px.histogram(x = grouped_data, nbins = 14, title = 'Mean Product per Salary Ranges')
       col1.plotly_chart(fig)

  
if page == 'Expense-Revenue Optimization':
  ## Salary Optimization
  slist = ['Covariance Analysis', 'Historical Trends']
  selection = st.selectbox("Select a Subject:", slist)

  if selection == 'Covariance Analysis':
    col1,col2 = st.columns(2)
    fig = px.imshow(cov_matrixQPEA, x=cov_matrixQPEA.columns, y=cov_matrixQPEA.columns)
    col1.plotly_chart(fig)
    fig = px.imshow(cov_matrixOCEA, x=cov_matrixOCEA.columns, y=cov_matrixOCEA.columns)
    col2.plotly_chart(fig, use_container_width = True)

  else:
    figure = px.line(dfq1, x=dfq1['paymentDate'], y=dfq1['amount'])
    labels = label_graph(figure, config)
    dfz1['label'] = labels['series']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Line(x=dfq1['paymentDate'], y=dfq1['amount'], name='Profit-Time'), secondary_y=False)
    fig.add_trace(go.Line(x=dfq1['paymentDate'], y=dfq1['amount'], name='label'), secondary_y=True)
    st.write(fig)


if page == 'Profit Analysis':
  col1,col2 = st.columns(2)
  fig = px.bar(x = ["revenue", "expenses"], y = [revenue,expenses], title = "Expenses v. Revenue")
  
  col1.plotly_chart(fig)
  fig = px.histogram(orderDetails, x="expenses", title ='Distribution of Expenses')
  
  col2.plotly_chart(fig, use_container_width = True)
