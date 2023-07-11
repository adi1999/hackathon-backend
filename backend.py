from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import h5py 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
# from keras.layers import Dense, Activation, Dropout
# from keras.layers import LSTM
# from keras.models import Sequential
import time
from sklearn.utils import shuffle
import onnxruntime

app = Flask(__name__)


def predict_load(timestamp):
    # Replace this with your ML program logic to predict load
    # Here, we are just returning a dummy list of load values for the next 12 hours
    load_predictions = []
    current_time = datetime.fromtimestamp(timestamp)
    for i in range(12):
        future_time = current_time + timedelta(hours=i+1)
        load_predictions.append({
            'timestamp': future_time.timestamp(),
            'load': i+1  # Dummy load value, replace with actual prediction
        })
    return load_predictions

    

def applyPCA(df):
    # Taking the full dataframe except the last column

    print(df.shape)
    df = df.iloc[:, 1:]

    # Subtracting the CPU,Disk,Netpacket mean column values from all rows from their respective columns
    df = df.sub(df.mean(axis=0), axis=1)

    # Converting the full dataframe into a matrix
    df_mat = np.asmatrix(df)
    

    # Get covariance matrix from dataframe matrix
    sigma = np.cov(df_mat.T)

    # Extract Eigen Values and Vectors
    eigVals, eigVec = np.linalg.eig(sigma)
    sorted_index = eigVals.argsort()[::-1] 
    eigVals = eigVals[sorted_index]
    eigVec = eigVec[:,sorted_index]
    eigVec = eigVec[:,:1]

    # Get transformed matrix
    transformedMatrix = df_mat.dot(eigVec)
    
    return np.array(transformedMatrix).flatten() 

def utilisation(df):

    print("started preprocessing")

    

    timestamps=df['CPU 1 YBLPVDAKDLWAPP1']
    swap_space_unused=df['swaptotal_mem'] -df['swapfree_mem']+df['inactive_mem']
    RAM_unused=df['memtotal_mem']-(df['memfree_mem']+df['buffers_mem']+df['cached_mem'])
    RAM=np.log(swap_space_unused)+np.log(RAM_unused)

    sr0=df['sr0_diskbusy']*df['sr0_diskbsize']
    sr0=sr0/100

    sda=df['sda_diskbusy']*df['sda_diskbsize']
    sda=sda/100

    sda1=df['sda1_diskbusy']*df['sda1_diskbsize']
    sda1=sda1/100

    sda2=df['sda2_diskbusy']*df['sda2_diskbsize']
    sda2=sda2/100

    sdb=df['sdb_diskbusy']*df['sdb_diskbsize']
    sdb=sdb/100

    sdb1=df['sdb1_diskbusy']*df['sdb1_diskbsize']
    sdb1=sdb1/100

    sdd=df['sdd_diskbusy']*df['sdd_diskbsize']
    sdd=sdd/100

    sdd1=df['sdd1_diskbusy']*df['sdd1_diskbsize']
    sdd1=sdd1/100

    sdc=df['sdc_diskbusy']*df['sdc_diskbsize']
    sdc=sdc/100

    dm_0=df['dm-0_diskbusy']*df['dm-0_diskbsize']
    dm_0=dm_0/100

    dm_1=df['dm-1_diskbusy']*df['dm-1_diskbsize']
    dm_1=dm_1/100

    dm_2=df['dm-2_diskbusy']*df['dm-2_diskbsize']
    dm_2=dm_2/100

    dm_3=df['dm-3_diskbusy']*df['dm-3_diskbsize']
    dm_3=dm_3/100

    dm_4=df['dm-4_diskbusy']*df['dm-4_diskbsize']
    dm_4=dm_4/100

    dm_5=df['dm-5_diskbusy']*df['dm-5_diskbsize']
    dm_5=dm_5/100

    dm_6=df['dm-6_diskbusy']*df['dm-6_diskbsize']
    dm_6=dm_6/100

    dm_7=df['dm-7_diskbusy']*df['dm-7_diskbsize']
    dm_7=dm_7/100

    dm_8=df['dm-8_diskbusy']*df['dm-8_diskbsize']
    dm_8=dm_8/100

    sd_total=sr0+sda+sda1+sda2+sdb+sdb1+sdd+sdd1+sdc+dm_0+dm_1+dm_2+dm_3+dm_4+dm_4+dm_5+dm_6+dm_7+dm_8

    read = (df['sr0_diskread']+df['sda1_diskread']+df['sda1_diskread']
    +df['sda2_diskread']+df['sdb_diskread']+df['sdb1_diskread']
            +df['sdd_diskread']+df['sdd1_diskread']+df['sdc_diskread']
            +df['dm-0_diskread']+df['dm-1_diskread']+df['dm-2_diskread']
                +df['dm-3_diskread']+df['dm-4_diskread']+df['dm-5_diskread']
            +df['dm-6_diskread']+df['dm-7_diskread']+df['dm-8_diskread'])

    write= (df['sr0_diskwrite']+df['sda1_diskwrite']+df['sda1_diskwrite']
    +df['sda2_diskwrite']+df['sdb_diskwrite']+df['sdb1_diskwrite']
            +df['sdd_diskwrite']+df['sdd1_diskwrite']+df['sdc_diskwrite']
            +df['dm-0_diskwrite']+df['dm-1_diskwrite']+df['dm-2_diskwrite']
                +df['dm-3_diskwrite']+df['dm-4_diskwrite']+df['dm-5_diskwrite']
            +df['dm-6_diskwrite']+df['dm-7_diskwrite']+df['dm-8_diskwrite'])

    transfer= (df['sr0_diskxfer']+df['sda1_diskxfer']+df['sda1_diskxfer']
    +df['sda2_diskxfer']+df['sdb_diskxfer']+df['sdb1_diskxfer']
            +df['sdd_diskxfer']+df['sdd1_diskxfer']+df['sdc_diskxfer']
            +df['dm-0_diskxfer']+df['dm-1_diskxfer']+df['dm-2_diskxfer']
                +df['dm-3_diskxfer']+df['dm-4_diskxfer']+df['dm-5_diskxfer']
            +df['dm-6_diskxfer']+df['dm-7_diskxfer']+df['dm-8_diskxfer'])
    
    disk=(read+write)/transfer

    netpacket=df['eth1-read-KB/s_net']+df['eth1-read/s_netpacket']+df['eth1-write-KB/s_net']+df['eth1-write/s_netpacket']
    for cpuNumber in range(1,18):
        df.drop(['Idle%_CPU' + str(cpuNumber)],axis=1,inplace=True)
    
    Cpu=np.log(np.sum(df.iloc[:,1:52],axis=1))

    DISK=pd.DataFrame({'disk':disk})

    NETPACKET=pd.DataFrame({'netpacket':netpacket})

    RAM=pd.DataFrame({'RAM':RAM})

    CPU=pd.DataFrame({'cpu':Cpu})

    df=pd.concat([timestamps,CPU,DISK,NETPACKET,RAM],axis=1)

    print("completed preprocessing")

    # print("apply pca")

    # combinedSystemLoad = applyPCA(df)

    # df['Combined System Load'] = pd.Series(combinedSystemLoad)

    # print("completed pca")

    return df


def predict_fault(data):
    # Replace this with your ML program logic to predict system faults
    # Here, we are just returning a dummy prediction and factors causing the fault
    
    preprocessedDf = utilisation(data)
    X = preprocessedDf[['cpu', 'disk', 'netpacket', 'RAM']]
    X = X.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    onnx_model_path = 'random_forest.onnx'
    sess = onnxruntime.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y = sess.run([output_name], {input_name: X.astype(np.float32)})[0]
    
    print("completed prediction")
    if (y[0]):
        prediction = 'failure'
    else:
        prediction = 'stable'
   
    return prediction


@app.route('/system-fault-prediction', methods=['POST'])
def system_fault_prediction():
    file = request.files.get('csv_file')
    if file is None:
        return jsonify({'error': 'CSV file not provided'}), 400

    try:
        data = pd.read_csv(file)
        prediction = predict_fault(data)
        return jsonify({'prediction': prediction})
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty CSV file'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Invalid CSV format'}), 400

@app.route('/load-prediction', methods=['POST'])
def load_prediction():
    timestamp = request.json.get('timestamp')
    if timestamp is None:
        return jsonify({'error': 'Timestamp not provided'}), 400

    try:
        timestamp = float(timestamp)
        load_predictions = predict_load(timestamp)
        return jsonify(load_predictions)
    except ValueError:
        return jsonify({'error': 'Invalid timestamp format'}), 400



if __name__ == '__main__':
    app.run()






