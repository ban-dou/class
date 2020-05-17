import numpy as np
import matplotlib.pyplot as plt

def data_generate(data_length):
    m1, m2 = [3, 1], [1, 3]
    sigma1, sigma2 = [[1,2],[2,5]], [[1,2],[2,5]]

    values1 = np.random.multivariate_normal(m1, sigma1, data_length)
    values2 = np.random.multivariate_normal(m2, sigma2, data_length)

    return values1, values2

def lda(values1, values2):
    n1 = len(values1)
    n2 = len(values2)
    #Mean
    ave1 = np.array([np.sum(values1[:,0])/n1, np.sum(values1[:,1])/n1])
    ave2 = np.array([np.sum(values2[:,0])/n2, np.sum(values2[:,1])/n2])
    #Variance within-class scatter
    s_w1 = np.array([[0,0],[0,0]], dtype=np.float)
    s_w2 = np.array([[0,0],[0,0]], dtype=np.float)
    for i in range(len(values1)):
        s_w1 += np.dot((values1[i]-ave1).reshape(2,1)\
        ,(values1[i]-ave1).reshape(1,2))
    for i in range(len(values2)):
        s_w2 += np.dot((values2[i]-ave2).reshape(2,1)\
        ,(values2[i]-ave2).reshape(1,2))
    s_w = s_w1 + s_w2
    #Linear transformation
    w = np.dot(np.linalg.inv(s_w), ave1 - ave2)
    #Draw the calculated axis
    x_lda = [i/10.0 for i in range(-100,100)]
    y_lda = [(w[1]/w[0])*x_lda[i] for i in range(len(x_lda))]

    plt.axes().set_aspect('equal', 'datalim')
    plt.scatter(values1[:,0], values1[:,1], s=10)
    plt.scatter(values2[:,0], values2[:,1], s=10)
    plt.plot(x_lda, y_lda, color='red')
    plt.title("LDA")
    plt.show()

    projection1 = np.dot(w,values1.T)
    projection2 = np.dot(w,values2.T)
    plt.hist(projection1, bins=30, ec='black')
    plt.hist(projection2, bins=30, ec='black')
    plt.title("The 1-d histograms of the sample data(LDA")
    plt.show()

def main():

    values1, values2 = data_generate(1000)
    lda(values1, values2)

if __name__ == '__main__':
    main()
