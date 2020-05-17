import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num):
    data = [0]*num
    data[0], data[1] = 3.0, 1.0
    a_1, a_2 = 0.5, 0.4

    for i in range(num-2):
        data[i+2] = a_1*data[i+1]+a_2*data[i]+np.random.randn()

    return data

def read_file():
    csv = pd.read_csv('canada.csv', header=0)
    return csv.values

def autocovariance(data, delay):
    ave = np.average(data)
    gamma = 0
    for i in range(delay,len(data)):
       gamma += float(data[i]-ave)*float(data[i-delay]-ave)
    return gamma/len(data)

class ar():
    def __init__(self, data, tr_rate=0.8, dim=2):
        self.data = data
        self.data_length = len(self.data)
        self.tr_rate = tr_rate
        self.dim = dim

    def spilt(self):
        self.split_num = int(self.data_length*(1-self.tr_rate))
        self.train = self.data[:-self.split_num]
        self.test = self.data[-self.split_num:]

    def fit(self):
        A = [[0] * self.dim for i in range(self.dim)]
        gamma = [autocovariance(self.train,i) for i in range(self.dim+1)]

        for i in range(self.dim):
            for j in range(self.dim):
                A[i][j] = gamma[abs(i-j)]
        b = [gamma[i] for i in range(1, self.dim+1)]

        self.param = np.dot(np.linalg.inv(A), b)

        self.sigma = 0
        for i in range(self.dim):
            self.sigma += self.param[i]*gamma[i+1]
        self.sigma = np.sqrt(gamma[0]-self.sigma)

    def predict(self):
        self.result = self.train

        for i in range(self.split_num):
            result = 0
            for j in range(self.dim):
                result += self.param[j]*self.result[-j-1]
            result += np.random.normal(0, self.sigma)
            np.append(self.result,result)

    def plot(self, title ="Data"):
        x = [i for i in range(self.data_length)]

        plt.title(title)
        plt.plot(x[-self.split_num-200:], self.data[-self.split_num-200:], label="True")
        plt.plot(x[-self.split_num:], self.result[-self.split_num:], label="Predict")
        vmax = max(max(self.data),max(self.result))
        vmin = min(min(self.data),min(self.result))
        plt.vlines(x[-self.split_num], vmin, vmax, "red", linestyles='dashed')
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend()
        plt.show()

def main():
    #Artificial data
    data = generate_data(1000)
    ar1 = ar(data, 0.8, 2)
    ar1.spilt()
    ar1.fit()
    ar1.predict()
    ar1.plot("Artificail Data")


    #Economic data
    data = read_file()
    ar2 = ar(data[:,3], 0.8, 2)
    ar2.spilt()
    ar2.fit()
    ar2.predict()
    ar2.plot("Economic Data")

if __name__ == '__main__':
    main()
