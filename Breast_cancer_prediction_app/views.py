from django.shortcuts import render
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


model = load('./savedModels/model.joblib')
standardise =load('./savedModels/scaler.joblib')
#print(model)
def predictor(request):
    return render(request, 'main.html')

def formInfo(request):
    mean_radius = request.GET['mean_radius']
    mean_texture = request.GET['mean_texture']
    mean_perimeter = request.GET['mean_perimeter']
    mean_area = request.GET['mean_area']
    mean_smoothness = request.GET['mean_smoothness']
    df2 = np.array([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness])
   # b = np.array(a, dtype = float)
   # c = [float(i) for i in a]
    # df2 = df2.astype('float64')
    
   # names= ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']
    #df2.columns = names
    df2 = np.reshape(df2,(1, -1))
    df2= standardise.transform(df2)
    print(df2)
    #testing_features=df2[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
    #df2[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
   # testing_features_standardised= standardise.fit_transform(df2)
   # print(testing_features_standardised)
   # y_pred = model.predict(testing_features_standardised)
    y_pred = model.predict(df2)
    print(y_pred)
    if y_pred[0]==0:
        y_pred = 'malignant'
    else:
        y_pred = 'Benign' 
    return render(request, 'result.html',{'result': y_pred})