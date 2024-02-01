import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# dataColumns = ["Close","Volume","mom","mom1","mom2","mom3","ROC_5","ROC_10","ROC_15","ROC_20","EMA_10","EMA_20",
#                "EMA_50","EMA_200","DTB4WK","DTB3","DTB6","DGS5","DGS10","Oil","Gold","DAAA","DBAA","GBP","JPY",
#                "CAD","CNY","AAPL","AMZN","GE","JNJ","JPM","MSFT","WFC","XOM","FCHI","FTSE","GDAXI","GSPC","HSI",
#                "IXIC","SSEC","RUT","NYSE","TE1","TE2","TE3","TE5","TE6","DE1","DE2","DE4","DE5","DE6","CTB3M",
#                "CTB6M","CTB1Y","AUD","Brent","CAC-F","copper-F","WIT-oil","DAX-F","DJI-F","EUR","FTSE-F",
#                "gold-F","HSI-F","KOSPI-F","NASDAQ-F","GAS-F","Nikkei-F","NZD","silver-F","RUSSELL-F","S&P-F",
#                "CHF","Dollar index-F","Dollar index","wheat-F","XAG","XAU"]

def loadData(indexType):
    fileName = f'../data/Processed_{indexType}.csv'
    data = pd.read_csv(fileName,chunksize=500,skiprows=-10,index_col=False)
    for idx,chunk in enumerate(data):
        if idx==0:
            df = pd.DataFrame(chunk)
        else:
            df = pd.concat([df,pd.DataFrame(chunk)])
    closeData = np.array(df["Close"]).reshape(-1,1)
    shiftData = closeData[10:,0]
    closeData = closeData[:-10,0]
    df = df.iloc[:-10]
    df["Next10DayReturn"] = 100 * (shiftData - closeData)/closeData
    return df

def getBestModel(XTrain,yTrain):
    searchParams = {
        'n_estimators': np.arange(10, 20, 100),
        }
    model = rf(random_state=42)
    modelGridSearch = ms.GridSearchCV(model, param_grid=searchParams,
                                    cv=5,n_jobs=-1)
    modelGridSearch.fit(XTrain,yTrain)
    return modelGridSearch

def printMSE(modelGridSearch,XTrain,XTest,yTrain,yTest):
    yTrainPredict = modelGridSearch.predict(XTrain)
    trainMSE = mean_squared_error(yTrain,yTrainPredict)
    yTestPredict = modelGridSearch.predict(XTest)
    testMSE = mean_squared_error(yTest,yTestPredict)
    print("train MSE =",trainMSE)
    print("test MSE =",testMSE)

def createImportanceDF(featureImportances,columns):
    importanceDFDictionary = {
        "columns": columns,
        "featureImportanceScores":featureImportances
    }
    importanceDF = pd.DataFrame(importanceDFDictionary,
                                columns=["columns","featureImportanceScores"]
                                ).sort_values(by="featureImportanceScores",ascending=False)
    return importanceDF

def displayScatterOfTop10Features(data,importanceDF,showTimeSeries=True):
    mostImportantFeatures = list(importanceDF.head(10)["columns"])
    for feature in mostImportantFeatures:
        r,p = spearmanr(data[[feature]].values.ravel(),data[["Next10DayReturn"]].values.ravel())
        print(f'Spearman correlation for Next 10 day return vs {feature} is r={r},p={p}')
        if(showTimeSeries):
            scaler = StandardScaler()
            timeAnalysis = data[[feature,"Next10DayReturn"]]
            timeAnalysis[[f'scaled {feature}',"scaled Next10DayReturn"]] = scaler.fit_transform(timeAnalysis)
            timeAnalysis = timeAnalysis.drop([feature,"Next10DayReturn"],axis=1)
            fig=plt.figure()
            timeAnalysis.plot.line()
            plt.show()

        fig = plt.figure()
        plt.scatter(data[feature],data["Next10DayReturn"])
        plt.xlabel(feature)
        plt.ylabel("Next 10 Day Return (%)")
        plt.title(f"Next 10 Day Return (%) vs {feature}")
        plt.show()

def retriveLowPValueColumns(X,y):
    goodColumns = []
    for col in list(X.columns):
        r,p = spearmanr(X[[col]].values.ravel(),y)
        if p < 0.05:
            goodColumns.append(col)
    return goodColumns

