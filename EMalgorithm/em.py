import numpy as np
import csv
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def input():
    with open("x.csv") as f:
        reader = csv.reader(f)
        list = [row for row in reader]
    return np.array(list, dtype='float')

def output_z(data):
    with open("z_em.csv", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in range(len(data)):
            writer.writerow([str(data[i][j]) for j in range(len(data[i]))])


def output_params(pi, mu, sigma):
    with open("params_em.dat", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Pi'])
        writer.writerow([str(pi[i]) for i in range(len(pi))])
        writer.writerow(['Mu'])
        for i in range(len(mu)):
            writer.writerow([str(mu[i][j]) for j in range(len(mu[i]))])
        writer.writerow(['Sigma'])
        for i in range(len(sigma)):
            writer.writerow(['Covariance matrix (claa'+str(i+1)+')'])
            for j in range(len(sigma[i])):
                writer.writerow([str(sigma[i][j][k]) for k in range(len(sigma[i][j]))])

class EmAlgorithm():
    def __init__(self, num_class, ite, data):
        #parameter
        self.pi = np.full(num_class, 1/float(num_class))
        self.mu = 8*np.random.rand(num_class, 3)-4
        self.sigma = np.array([np.diag(np.random.rand(3)+1)]*num_class)
        self.gamma = np.full((num_class, len(data)), 1/float(num_class))
        self.likelihood = 0

        self.data = data
        self.num_class = num_class
        self.num_data = len(data)
        self.ite = ite


    def E_step(self):
        deno = np.zeros(self.num_data)
        nume = np.zeros((self.num_class,self.num_data))
        for k in range(self.num_class):
            tmp = self.pi[k]*st.multivariate_normal.pdf(self.data, self.mu[k], self.sigma[k])
            deno += tmp
            nume[k] = tmp

        for k in range(self.num_class):
            self.gamma[k] = nume[k]/deno

        self.loglikelihood = np.sum(np.log(deno))

    def M_step(self):
        s_k_1 = np.sum(self.gamma, axis = 1)
        s_1 = np.sum(s_k_1)
        s_k_x = np.dot(self.gamma, self.data)

        self.pi = s_k_1/s_1
        for k in range(self.num_class):
            self.mu[k] = s_k_x[k]/s_k_1[k]

        s_k_xx = np.array([np.diag(np.zeros(3))]*self.num_class)
        for k in range(self.num_class):
            tmp = np.einsum('i,ij->ij', self.gamma[k], self.data)
            s_k_xx[k] = np.einsum('ij,ik->ijk', tmp, self.data).sum(axis=0)
            self.sigma[k] = s_k_xx[k]/s_k_1[k]-np.dot(np.array([self.mu[k]]).T,np.array([self.mu[k]]))

    def em(self):
        pi = np.zeros((self.ite, self.num_class))
        mu = np.zeros((self.ite, self.num_class, 3))
        sigma = np.zeros((self.ite, self.num_class, 3, 3))
        gamma = np.zeros((self.ite, self.num_class, len(self.data)))
        loglikelihood = np.zeros(self.ite)
        for i in range(self.ite):
            self.E_step()
            self.M_step()
            pi[i]=self.pi
            mu[i]=self.mu
            sigma[i]=self.sigma
            gamma[i]=self.gamma
            loglikelihood[i]=self.loglikelihood
        for k in range(self.num_class):
            self.sigma[k] = np.linalg.inv(self.sigma[k])
        return pi,mu,sigma,gamma,loglikelihood

def plot3D_class_4(gamma, data):
    a = np.argmax(gamma, 0)
    x1,x2,x3,x4=[],[],[],[]
    for i in range(len(a)):
        if(a[i]==0):
            x1.append(data[i])
        elif(a[i]==1):
            x2.append(data[i])
        elif(a[i]==2):
            x3.append(data[i])
        elif(a[i]==3):
            x4.append(data[i])
    x1=np.array(x1)
    x2=np.array(x2)
    x3=np.array(x3)
    x4=np.array(x4)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[:,0], x1[:,1], x1[:,2],c='red')
    ax.scatter(x2[:,0], x2[:,1], x2[:,2],c='blue')
    ax.scatter(x3[:,0], x3[:,1], x3[:,2],c='black')
    ax.scatter(x4[:,0], x4[:,1], x4[:,2],c='yellow')
    plt.show()


def plot_loglikelihood(data):
    x = [i+1 for i in range(len(data))]
    plt.figure()
    plt.plot(x, data, marker="o")
    plt.xlabel("Iterations")
    plt.ylabel("Log likelihood")
    plt.show()

def main():
    data = input()
    em = EmAlgorithm(4, 100, data)
    pi,mu,sigma,gamma,loglikelihood = em.em()
    print loglikelihood
    plot3D_class_4(gamma[-1], data)
    plot_loglikelihood(loglikelihood)
    output_z(gamma[-1].T)
    output_params(pi[-1],mu[-1].T,sigma[-1])

if __name__ == '__main__':
    main()
