from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import chart_studio.plotly as py
import mplcyberpunk
# from darts import TimeSeries

print("before")
model_dict= pickle.load(open(r"pickle/best_mse_dict.pkl","rb"))
app = Flask(__name__)
VALID_ID_MSG = "Error: Given ID is incorrect, Please enter a valid ID"
NOT_ENOUGHT_MSG ="Note: Given ID does not enough quarter data to forecast results"
MEDICARE="Medicare"
COMMERCIAL="Commercial"
BOTH="Medicare, Commercial"
LAB='Lab'
PHYSICIAN='Physician'
FORECAST='Forecast'
forecast_dates = None
forecast_len = 0
forecast_model=None

def getForecastDatesLength(file):
    rand_id = 56227
    data_copy = pd.read_csv(f"data/{file}.csv",header=0,index_col=0)
    medi = data_copy[(data_copy.id == rand_id) & (data_copy.type == 'M')]
    comm = data_copy[(data_copy.id == rand_id) & (data_copy.type == 'C')]
    global forecast_dates
    global forecast_len
    forecast_len = len(comm) - len(medi)
    start_date = medi.index[-1:][0]
    forecast_dates = list(pd.date_range(start=start_date,periods=forecast_len+1,
                                        freq='QS-OCT').astype(str)[1:])
    
def Checkless50Data(id):
    id_list = model_dict.keys()
    if (id not in id_list):
        return True
    return False

def getIDSourceType(id,data):
    src = np.unique(data[(data.id == id)]['source'])
    types = np.unique(data[(data.id == id)]['type'])
    src = LAB if 'L' in src else PHYSICIAN
    types = COMMERCIAL if 'M' not in types else MEDICARE if 'C' not in types else BOTH
    return src,types

# def convertTsToDf(ts):
#     if (ts != None ):
#         df = TimeSeries.pd_dataframe(ts)
#         df['volume'] = np.round(df['volume']).astype('int')
#         df[df['volume'] < 0] = 1
#         return df
#     return ts

def getData(id_model,model_type):
    best_model = id_model['best_model']
    med_data = id_model['med_data']
    com_data = id_model['com_data']
    feature = None
    if(model_type not in ['NBEATSI','NBEATSS','RNN']):
        med_data = pd.DataFrame(med_data)
        com_data = pd.DataFrame(com_data)
        if (not com_data.empty):
            feature = com_data[len(med_data):]
    else:
        if (com_data != None):
            feature = com_data[len(med_data):]
            
    return best_model,med_data,com_data,feature  

def predictAutoArima(id_model,src,types,model_type):
    print("It is AUTO ARIMA")
    best_model,med_data,com_data,feature = getData(id_model,model_type)
    predict_df = pd.DataFrame(best_model.predict(X=pd.DataFrame(feature),
                                          n_periods=forecast_len), 
                                          index= forecast_dates,
                                          columns = ['volume'])
    print(predict_df)
    predict_df = pd.DataFrame(predict_df)
    predict_df['volume'] = np.round(predict_df['volume']).astype('int')
    return med_data,com_data,predict_df


def predictProphet(id_model,src,types,model_type):
    print("It is PROPHET")
    best_model,med_data,com_data,feature = getData(id_model,model_type)
    med_data = pd.DataFrame(med_data)
    com_data = pd.DataFrame(com_data)
    feature = pd.DataFrame(feature)
    forecast = best_model.make_future_dataframe(periods=forecast_len, 
                                                freq='Q',
                                                include_history = False)
    forecast['C_volume']= None if types != BOTH else feature['volume'].reset_index(drop=True)
    predict = best_model.predict(forecast)
    predict_df = pd.DataFrame(np.round(predict['yhat']).astype('int'))
    predict_df.columns = ['volume']
    predict_df.index = forecast_dates
    predict_df[predict_df['volume'] < 0 ] = 1
    return med_data,com_data,predict_df


def predictNbeatsI(id_model,src,types,model_type):
    print("It is NBEATSI")
    best_model,med_data,com_data,feature = getData(id_model,model_type)
    predict = best_model.predict(n=forecast_len,
                           series = med_data,
                           past_covariates=com_data,
                           verbose=False,
                           n_jobs=-1)
    # med_data = convertTsToDf(med_data)
    # com_data = convertTsToDf(com_data)
    # predict = convertTsToDf(predict)
    return med_data,com_data,predict                                                                                                  


def predictNbeatsS(id_model,src,types,model_type):
    print("It is NBEATSS")
    best_model,med_data,com_data,feature = getData(id_model,model_type)
    predict = best_model.predict(n=forecast_len,
                           series = med_data,
                           past_covariates=com_data,
                           verbose=False,
                           n_jobs=-1)
    # med_data = convertTsToDf(med_data)
    # com_data = convertTsToDf(com_data)
    # predict = convertTsToDf(predict)
    return med_data,com_data,predict


def predictRNN(id_model,src,types,model_type):
    print("It is RNN")
    best_model,med_data,com_data,feature = getData(id_model,model_type)
    predict = best_model.predict(n=forecast_len,
                           series = med_data,
                           future_covariates =com_data,
                           verbose=False,
                           n_jobs=-1)
    # med_data = convertTsToDf(med_data)
    # com_data = convertTsToDf(com_data)
    # predict = convertTsToDf(predict)
    return med_data,com_data,predict
    
    
def getPredictResult(id,src,types):
    id_model = model_dict[id]
    model_type = id_model['model_type']
    global forecast_model
    forecast_model = model_type
    if model_type == 'AUTO_ARIMA':
        predict = predictAutoArima(id_model,src,types,model_type)
    elif model_type == 'PROPHET':
        predict = predictProphet(id_model,src,types,model_type)
    elif model_type == 'NBEATSI':
        predict = predictNbeatsI(id_model,src,types,model_type)
    elif model_type == 'NBEATSS':
        predict = predictNbeatsS(id_model,src,types,model_type)
    else:
        predict = predictRNN(id_model,src,types,model_type)     
    return predict

def plotResult(med_data,com_data,src,types,pred_result,id):
    plt.figure(figsize=(12,5))
    plt.style.use("cyberpunk")
    fig= go.Figure()
    dummy = pd.concat([med_data[-1:],pred_result])
    com_plot=True
    print("Plotting Medicare")  
    fig.add_trace(go.Scatter(x=med_data.index,
                    y=med_data['volume'],
                    name='Medicare',
                    mode='lines+markers',
                    marker=dict(color='#bb2c12',size=8)))

    if (forecast_model in ['NBEATSI','NBEATSS','RNN']) and (com_data==None):
        com_plot = False
    
    if (forecast_model not in ['NBEATSI','NBEATSS','RNN']) and (com_data.empty):
        com_plot = False
    print(com_plot)
    if (com_plot):
        fig.add_trace(go.Scatter(x=com_data.index,
                      y=com_data['volume'],
                      mode='lines+markers',
                      name='Commercial',
                      marker=dict(color='#08f6fd',size=8))) 
        
    print("Plotting lines1")    
    fig.add_trace(go.Scatter(x=dummy.index,
                  y=dummy['volume'],
                  mode='lines',
                  showlegend = False,
                  hovertemplate=None,
                  hoverinfo='skip',
                  line=dict(color='#F5D300',dash='dash'))) 
    print("Plotting lines2")  
    fig.add_trace(go.Scatter(x=pred_result.index,
                  y=pred_result['volume'],
                  mode='lines+markers',
                  name='Medicare Forecast',
                  marker=dict(color='#F5D300',size=8))) 
    
    fig.update_layout(plot_bgcolor='#212946',
                              paper_bgcolor='#212946',
                              title=f"Forcast for {src} ID {id}",
                              yaxis_title="Patient Volume",
                              font_color='#818594',
                              hovermode="x")
    
    fig.update_xaxes(showgrid=True, gridwidth=0.01, gridcolor='#435266')
    fig.update_yaxes(showgrid=True, gridwidth=0.01, gridcolor='#435266')
    py.plot(fig, filename="forecast_med",default=False)
    print("done with plotting")


        
def predictResult(id,file):
    data = pd.read_csv(f"data/{file}.csv",header=0,index_col=0)
    src,types = getIDSourceType(id,data)
    print(src)
    print(types)
    med_data,com_data,pred_result = getPredictResult(id,src,types)
    print(pred_result)
    print("Plotting")
    plotResult(med_data,com_data,src,types,pred_result,id)
    return src,types
    

    
def invalidID(ID,file):
    data_copy = pd.read_csv(f"data/{file}.csv",header=0,index_col=0)
    id_list = np.unique(data_copy['id'])
    if ID not in id_list:
        return True
   


@app.route('/predict', methods=['POST'])
def predict():
    ID=request.form['ID']
    # ID=ID
    if (ID.isdigit()):
        ID=int(ID)
    print("Inside Predict")
    print (f"Id is {ID}")
    print("Validating the Data")
    if invalidID(ID,"data_copy1"):
        print("Rendering back to home")
        return render_template('home.html',result=VALID_ID_MSG)
    
    if invalidID(ID,"data_copy4"):
        print("Not enough data")
        return render_template('home.html',result=NOT_ENOUGHT_MSG)
        
    print("Forecasting")
    getForecastDatesLength("data_copy4")
    src,types = predictResult(ID,'data_copy4')
    info = f"Forecast for {forecast_len} Quarter"
    print("Done")
    return render_template('predict.html',ID=ID,info=info)
               
                           
@app.route('/')
def home():
    return render_template('home.html')

if __name__=="__main__":
    print("Inside Main")
    # predict('1134198989')
    app.run(debug=False)