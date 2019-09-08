import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_predict_real = 8
num_predict_art = 100
feature = 3

def data_generator(num):
    data = [0]*num
    data[0], data[1] = 3, 1
    a_1, a_2 = 0.5, 0.4

    for i in range(num-2):
        data[i+2] = a_1*data[i+1]+a_2*data[i]+np.random.randn()

    return data

def read_file():
    csv = pd.read_csv('canada.csv', header=0)
    return csv.values

def autocovariance_ar(data, delay):
    ave = np.average(data)
    gamma = 0
    for i in range(delay,len(data)):
       gamma += float(data[i]-ave)*float(data[i-delay]-ave)
    return gamma/len(data)

def ar(data):
    dim = 2
    A = [[0] * dim for i in range(dim)]
    gamma = [autocovariance_ar(data,i) for i in range(dim+1)]

    for i in range(dim):
        for j in range(dim):
            A[i][j] = gamma[abs(i-j)]
    b = [gamma[i] for i in range(1, dim+1)]

    param = np.dot(np.linalg.inv(A), b)

    sigma = gamma[0] - (param[0]*gamma[1]+param[1]*gamma[2])
    return param,  sigma

def predict_ar(data, param, num_predict):
    for i in range(num_predict):
        result = param[0][0]*data[-num_predict+i]+param[0][1]\
        *data[-num_predict+i-1]+np.random.normal(0, param[1])
        data = np.append(data, result)
    return data

def main():
    # artificial data
    data = data_generator(1000)
    train = data[:-num_predict_art]
    test = data[-num_predict_art:]

    param = ar(train)
    print "Parameter(artificial data)"+str(param[0])
    print "Variance(artificial data)"+str(param[1])
    result = predict_ar(train, param, num_predict_art)

    x = [i for i in range(1000)]

    plt.title("Artificial data")
    plt.plot(x[-200:], data[-200:], label="True data")
    plt.plot(x[-200:], result[-200:], label="Predict")
    plt.vlines(899, -4, 4, "red", linestyles='dashed')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # real data
    data = read_file()
    train = data[:-num_predict_real]
    test = data[-num_predict_real:]

    param = ar(train[:,feature])
    print "Parameter(economic data)"+str(param[0])
    print "Variance(economic data)"+str(param[1])
    result = predict_ar(train[:,feature], param, num_predict_real)

    x = data[:,0]+[(i%4)*0.25 for i in range(len(data[:,0]))]

    plt.title("Real wage in economic data")
    plt.plot(x, data[:,feature], label="True data")
    plt.plot(x, result, label="Predict")
    plt.vlines(1998.75, 100, 500, "red", linestyles='dashed')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
