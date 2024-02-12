'''
Stock Price Prediction for a particular stock
Author - Aditya Bhatt 10:30 AM 12-02-2024
Notes-
1.
'''
#Library Import
import numpy as np
import pandas as pd
import yfinance as yahooFinance
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")
#ETL
GetFacebookInformation = yahooFinance.Ticker("ZOMATO.NS") 
data=pd.DataFrame(GetFacebookInformation.history(period="max"))
#Dumping a small sample of the data
data.head(10).to_csv("zomato.csv")
data=data[["Open", "High"  , "Low",  "Close"  ,   "Volume"]]
data.reset_index(inplace=True)

#Seeing the results
print(data.columns)
print(data.head(1))