import socketio

sio = socketio.Client()

def connectToServer():
    sio.connect('http://localhost:5000', wait_timeout = 20)

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")
    print('Exiting the program!!!')
    exit(0)

@sio.on('frontend_hardware_output')
def data_processing(data):
    print(data)

@sio.on('frontend_anomaly_output')
def data_processing(data):
    print(data)

if __name__ == '__main__':
    connectToServer()