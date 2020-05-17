import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class_num = 2
file_name = 'iris'
attribute1 = 0
attribute2 = 2
dimension = 4
iteration = 10

def read_file(str, num1, num2):
    file = pd.read_csv('data/'+ str +'.data')
    file.to_csv('data/'+ str +'.data.csv',index=False,header=None)
    csv = pd.read_csv('data/'+ str +'.data.csv',header=None)
    data = csv.iloc[:,[num1, num2]].values
    return data

def split_data(data, percent):
    return np.split(data, [int(len(data) * percent)])

def polynomial(x, i, m):
    return np.power(x, i)

def gaussian(x, i, m):
    mu = np.array([(j / float(m))+4 for j in range(0, m*2, 2)])
    if i == 0:
        return [1]*len(x)
    else:
        return np.exp(-1*np.power((x - mu[i]), 2))

def sigmoid(x, i, m):
    mu = np.array([(j / float(m))+4 for j in range(0, m*2, 2)])
    if i == 0:
        return [1]*len(x)
    else:
        return 1/(1+np.exp(-1*(x - mu[i])))

def weight_regression(data, f, m):
    x = np.array([f(data[:, 0], i, m) for i in range(m)])
    weight = np.dot(np.dot(np.linalg.inv(np.dot(x, x.T)), x), data[:,1])
    return weight

def predict_regression(weight, x, f):
    y = weight[0]
    for i in range(1, len(weight)):
        y += weight[i] * f(x, i, len(weight))
    return x, y

def reguration_polynomial_regression(data, m, lamb):
    x = np.array([np.power(data[:, 0], i) for i in range(m)])
    weight = np.dot(np.dot(np.linalg.inv(lamb*np.eye(m)+np.dot(x, x.T)), x), data[:,1])
    return weight

def rms(predict, data):
    return np.sqrt(np.sum(np.power(predict - data[:,1], 2))/len(data))

def plot_regression(train, test, predict, name):
    plt.scatter(train[:,0],train[:,1], label="Training")
    plt.scatter(test[:,0],test[:,1], label="Test")
    plt.plot(predict[0], predict[1], color='black')
    plt.title(name)
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")
    plt.legend()
    plt.show()

def regression():
    rms_polynomial, rms_sigmoid, rms_gaussian, rms_reguration_polynomial = [], [], [], []
    data = read_file(file_name, attribute1, attribute2)
    train_and_valid, test = split_data(data[0:49], 0.9)
    '''
    validation test
    '''
    for i in range(iteration):
        train_and_valid_sorted = np.random.permutation(train_and_valid)
        train, valid = split_data(train_and_valid_sorted, 0.9)

        x = [j/100.0 for j in range(400,600)]

        weight_polynomial = weight_regression(train, polynomial, dimension)
        valid_predict_polynomial = predict_regression(weight_polynomial, valid[:, 0], polynomial)

        weight_gaussian = weight_regression(train, gaussian, dimension)
        valid_predict_gaussian = predict_regression(weight_gaussian, valid[:, 0], gaussian)

        weight_sigmoid = weight_regression(train, sigmoid, dimension)
        valid_predict_sigmoid = predict_regression(weight_sigmoid, valid[:, 0], sigmoid)

        weight_reguration_polynomial = reguration_polynomial_regression(train, dimension, 1)
        valid_predict_reguration_polynomial = predict_regression(weight_reguration_polynomial, valid[:, 0], polynomial)

        if i == 0:

            predict_polynomial = predict_regression(weight_polynomial, test[:, 0], polynomial)
            predict_gaussian = predict_regression(weight_gaussian, test[:, 0], gaussian)
            predict_sigmoid = predict_regression(weight_sigmoid, test[:, 0], sigmoid)
            predict_reguration_polynomial = predict_regression(weight_reguration_polynomial, test[:, 0], polynomial)

            print "regression predict(polynomial)", predict_polynomial
            print "regression predict(gaussian)", predict_gaussian
            print "regression predict(sigmoid)", predict_sigmoid
            print "regression predict(reguration polynomial)", predict_reguration_polynomial

            plot_regression(train, test, predict_regression(weight_polynomial, x, polynomial), "Liner regression(polynomial)")
            plot_regression(train, test, predict_regression(weight_gaussian, x, gaussian), "Liner regression(gaussian)")
            plot_regression(train, test, predict_regression(weight_sigmoid, x, sigmoid), "Liner regression(sigmoid)")
            plot_regression(train, test, predict_regression(weight_reguration_polynomial, x, polynomial), "Regularization liner regression(polynomial)")

        rms_polynomial.append(rms(valid_predict_polynomial, valid))
        rms_gaussian.append(rms(valid_predict_gaussian, valid))
        rms_sigmoid.append(rms(valid_predict_sigmoid, valid))
        rms_reguration_polynomial.append(rms(valid_predict_reguration_polynomial, valid))

    erms_polynomial = np.average(rms_polynomial)
    erms_gaussian = np.average(rms_gaussian)
    erms_sigmoid = np.average(rms_sigmoid)
    erms_reguration_polynomial = np.average(rms_reguration_polynomial)

    print "regression varidation(polynomial)",erms_polynomial
    print "regression varidation(gaussian)",erms_gaussian
    print "regression varidation(sigmoid)",erms_sigmoid
    print "regression varidation(reguration polynomial)",erms_reguration_polynomial

def least_squares(data):
    x_temp = np.concatenate([data[0], data[1]])
    x = np.array([np.insert(x_temp[i], 0, 1) for i in range(len(x_temp))])
    t = np.concatenate([[[1,0] for i in range(len(data[0]))], [[0,1] for i in range(len(data[1]))]])
    weight = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), t)
    return weight

def perceptron(data):
    x_temp = np.concatenate([data[0], data[1]])
    x = np.array([np.insert(x_temp[i], 0, 1) for i in range(len(x_temp))])
    t = np.concatenate([[1]*len(data[0]), [-1]*len(data[1])])
    weight = [1.0, 1.0, 1.0]
    correct = 0
    while(correct<len(x)):
        correct = 0
        for i in range(len(x)):
            if np.dot(weight, x[i].T)*t[i] <= 0:
                weight += 0.1*x[i].T*t[i]
            else:
                correct += 1
    return weight

def rate_least_squares(weight, x):
    x0 = np.array([np.insert(x[0][i], 0, 1) for i in range(len(x[0]))])
    x1 = np.array([np.insert(x[1][i], 0, 1) for i in range(len(x[1]))])
    y0 = np.dot(weight.T, x0.T)
    y1 = np.dot(weight.T, x1.T)
    count = [0, 0]
    for i in range(len(y0[0])):
        if y0[0][i] > y0[1][i]:
            count[0] += 1
    for i in range(len(y1[0])):
        if y1[0][i] < y1[1][i]:
            count[1] += 1
    return np.sum(count)/float(len(x[0])+len(x[1]))

def rate_perceptron(weight, x):
    x0 = np.array([np.insert(x[0][i], 0, 1) for i in range(len(x[0]))])
    x1 = np.array([np.insert(x[1][i], 0, 1) for i in range(len(x[1]))])
    y0 = np.dot(weight.T, x0.T)
    y1 = np.dot(weight.T, x1.T)
    count = [0, 0]
    for i in range(len(y0)):
        if y0[i] >= 0:
            count[0] += 1
    for i in range(len(y1)):
        if y1[i] < 0:
            count[1] += 1
    return np.sum(count)/float(len(x[0])+len(x[1]))

def discriminant_least_squares(w):
    x = [i/100.0 for i in range(400,700)]
    y = [-1*(w.T[0][1]-w.T[1][1])*x[i]/(w.T[0][2]-w.T[1][2])-(w.T[0][0]-w.T[1][0])/(w.T[0][2]-w.T[1][2]) for i in range(len(x))]
    return [x, y]

def discriminant_perceptron(w):
    x = [i/100.0 for i in range(400,700)]
    y = [-1*(w[1]/w[2]) * x[i] -1*(w[0]/w[2]) for i in range(len(x))]
    return [x, y]

def plot_classification(w, train, test, f, name):
    plt.scatter(train[0][:,0], train[0][:,1], label="Training(iris setosa)")
    plt.scatter(train[1][:,0], train[1][:,1], label="Training(iris versicolor)")
    plt.scatter(test[0][:,0], test[0][:,1], label="Test(iris setosa)")
    plt.scatter(test[1][:,0], test[1][:,1], label="Test(iris versicolor)")
    predict = f(w)
    plt.plot(predict[0], predict[1], color='black')
    plt.title(name)
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")
    plt.legend(loc="lower right", prop={'size':7,})
    plt.show()



def classification():
    data = read_file(file_name, attribute1, attribute2)
    train_and_valid, test = [],[]
    for i in range(class_num):
        if i == 0:
            train_and_valid_temp, test_temp = split_data(data[0:49], 0.9)
        if i == 1:
            train_and_valid_temp, test_temp = split_data(data[49:99], 0.9)
        train_and_valid.append(train_and_valid_temp)
        test.append(test_temp)

    rate_least_squares_array = []
    rate_perceptron_array = []
    for i in range(iteration):
        train, valid = [], []
        for j in range(class_num):
            train_and_valid_sorted_temp = np.random.permutation(train_and_valid[j])
            train_temp, valid_temp = split_data(train_and_valid_sorted_temp, 0.9)
            train.append(train_temp)
            valid.append(valid_temp)

        weight_least_squares = least_squares(train)
        rate_least_squares_array.append(rate_least_squares(weight_least_squares, valid))

        weight_perceptron = perceptron(train)
        rate_perceptron_array.append(rate_perceptron(weight_perceptron, valid))

        if i == 0:
            plot_classification(weight_least_squares, train, test, discriminant_least_squares, "Classification(least squares)")
            plot_classification(weight_perceptron, train, test, discriminant_perceptron, "Classification(perceptron)")
    erate_least_squares = np.average(rate_least_squares_array)
    erate_perceptron = np.average(rate_perceptron_array)
    print "classification(least squares) erate", erate_least_squares
    print "classification(perceptron) erate", erate_perceptron

def main():
    regression()
    classification()

if __name__ == '__main__':
    main()
