import socketio
import csv
import time

sio = socketio.Client()

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error():
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")
    print('Exiting the program!!!')
    exit(0)


@sio.on('model_input')
def test_fun(data):
    print('Reached Here!!!')

def connectToServer():
    sio.connect('http://localhost:5000', wait_timeout = 20)
    start_data_streaming()

def start_data_streaming():
    count = 0
    with open('data_source/final_combined_system_metric_data.csv', newline='') as hardwareCsv , open('data_source/TrainDf.csv', newline='') as networkCsv:
        hardwareData = csv.reader(hardwareCsv, delimiter=',')
        networkData = csv.reader(networkCsv, delimiter=',')
        print("Something happened!!!")
        for (hardware, network) in zip(hardwareData, networkData):
            if(count==0):
                count+=1
                continue
            count+=1
            time.sleep(20)
            print({'hardwareData': 'Timestamp: ' + ", ".join(hardware),
                                        'networkData': ", ".join(network),
                                        'count': count})
            sio.emit('data_source', {'hardwareData': ",".join(hardware),
                                        'networkData': ",".join(network),
                                        'count': count})

if __name__ == '__main__':
    connectToServer()