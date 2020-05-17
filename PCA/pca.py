import numpy as np
import matplotlib.pyplot as plt

def data_generate(data_length):
    m1, m2 = [3, 1], [1, 3]
    sigma1, sigma2 = [[1,2],[2,5]], [[1,2],[2,5]]

    values1 = np.random.multivariate_normal(m1, sigma1, data_length)
    values2 = np.random.multivariate_normal(m2, sigma2, data_length)

    return values1, values2

def pca(values1, values2):
    values = np.concatenate([values1, values2])

    n = len(values)
    x = values[:,0]
    y = values[:,1]
    #Mean
    x_ave = np.sum(x)/n
    y_ave = np.sum(y)/n
    #Covariance matrix
    s11 = np.sum(np.power(x - x_ave,2))/n
    s22 = np.sum(np.power(y - y_ave,2))/n
    s12 = np.sum(np.dot(x - x_ave, y - y_ave))/n
    s = [[s11, s12],[s12, s22]]
    #Find eigenvalue and eigenvectors of covariance matrix
    eig, eig_vec= np.linalg.eig(s)
    #Draw the 1st principal axis
    x_pca = [i/10.0 for i in range(-10,50)]
    y_pca = [(eig_vec[1][np.argmax(eig)]/eig_vec[0][np.argmax(eig)])\
    *(x_pca[i]-x_ave)+y_ave for i in range(len(x_pca))]

    plt.axes().set_aspect('equal', 'datalim')
    plt.scatter(values[:,0], values[:,1], s=10)
    plt.plot(x_pca, y_pca, color='red')
    plt.title("PCA")
    plt.show()
    #Projection to the 1st principal axis
    projection = np.dot(eig_vec[:,np.argmax(eig)], values.T)
    plt.hist(projection, bins=30, ec='black')
    plt.title("The 1-d histograms of the sample data(PDA)")
    plt.show()

def main():

    values1, values2 = data_generate(1000)
    pca(values1, values2)

if __name__ == '__main__':
    main()
