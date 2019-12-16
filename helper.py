import numpy as np
import pandas as pd

def process_data(data, timePortion) :
    data = data.iloc[::-1]
    size = int(len(data) / len(data.columns))
    trainX = []
    trainY = [] 
    features = []
    days = []
    
    for i in range(0, size):
        v = float(data['Close'][i].replace('$','').replace(' ','')) 
        features.append(v)
        days.append(data['Date'][i])

    # Scale the values
    scaledData = minMaxScaler(features, np.min(features), np.max(features))
    scaledFeatures = scaledData["data"]
    try :
        for i in range(timePortion, timePortion + size) :
            for j in range(i - timePortion, i):
                    trainX.append(scaledFeatures[j])

            trainY.append(scaledFeatures[i])
    except Exception as ex:
        print(ex)

    return {
        "size": (size - timePortion),
        "timePortion": timePortion,
        "trainX": trainX,
        "trainY": trainY,
        "min": scaledData["min"],
        "max": scaledData["max"],
        "originalData": features,
        'date' : days
    }

def minMaxScaler (data, min, max) :
    scaled_data = (data - min) / (max - min)
    return {
        "data" : scaled_data,
        "min" : min,
        "max" : max
    }

def minMaxInverseScaler(data, min, max) :
    scaledData = data * (max - min) + min
    return {
        "data": scaledData,
        "min": min,
        "max": max
    }
def generate(filename, timeportion) :
    v = pd.read_csv(filename)
    ret = process_data(v, timeportion)
    print (ret)
    return ret
    
if __name__ == "__main__":
    v = generate('./appl.csv')
    print (v)
    