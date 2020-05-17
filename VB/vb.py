import numpy as np
import csv
import scipy.stats as st
from scipy.special import digamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def input():
    with open("x.csv") as f:
        reader = csv.reader(f)
        list = [row for row in reader]
    return np.array(list, dtype='float')

def output_z(data):
    with open("z_vb.csv", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in range(len(data)):
            writer.writerow([str(data[i][j]) for j in range(len(data[i]))])

def output_params(pi,sigma,alpha,beta,nu,w,m):
    with open("params_vb.dat", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Pi'])
        writer.writerow([str(pi[i]) for i in range(len(pi))])
        writer.writerow(['Sigma'])
        writer.writerow([str(sigma[i]) for i in range(len(sigma))])
        writer.writerow(['Alpha'])
        writer.writerow([str(alpha[i]) for i in range(len(alpha))])
        writer.writerow(['Beta'])
        writer.writerow([str(beta[i]) for i in range(len(beta))])
        writer.writerow(['Nu'])
        writer.writerow([str(nu[i]) for i in range(len(nu))])
        writer.writerow(['M'])
        for i in range(len(m)):
            writer.writerow([str(m[i][j]) for j in range(len(m[i]))])
        writer.writerow(['W'])
        for i in range(len(w)):
            writer.writerow(['(class'+str(i+1)+')'])
            for j in range(len(w[i])):
                writer.writerow([str(w[i][j][k]) for k in range(len(w[i][j]))])

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

class VariationalBayes():
    def __init__(self, num_class, ite, data):
        #parameter
        self.pi = np.full(num_class, 1/float(num_class))
        self.sigma_t = np.zeros(num_class)
        self.gamma = np.full((num_class, len(data)), 1/float(num_class))
        self.likelihood1 = 0
        self.likelihood2 = 0

        self.alpha = np.full(num_class,len(data)/num_class)
        self.alpha0 = np.ones(num_class)
        self.beta = np.full(num_class,len(data)/num_class)
        self.beta0 = 1.0
        self.nu = np.full(num_class,3)
        self.nu0 = 3
        self.w = np.tile(np.eye(3), (num_class, 1, 1)).T
        self.w0 = np.eye(3)
        self.m = 8*np.random.rand(num_class, 3)-4
        self.m0 = np.zeros(3)

        self.data = data
        self.num_class = num_class
        self.num_data = len(data)
        self.ite = ite

        self.s_k_1 = np.zeros(num_class)
        self.s_k_x = np.zeros((num_class, len(data)))
        self.s_k_xx = np.array([np.diag(np.zeros(3))]*num_class)

    def calc_statistics(self):
        self.s_k_1 = np.sum(self.gamma, axis = 1)
        s_1 = np.sum(self.s_k_1)
        self.s_k_x = np.dot(self.gamma, self.data)

        for k in range(self.num_class):
            tmp = np.einsum('i,ij->ij', self.gamma[k], self.data)
            self.s_k_xx[k] = np.einsum('ij,ik->ijk', tmp, self.data).sum(axis=0)

    def E_step(self):
        self.pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))

        self.sigma_t = np.exp(digamma(self.nu - np.arange(3)[:, None]).sum(axis=0) \
        + len(self.data[0]) * np.log(2) + np.linalg.slogdet(self.w.T)[1])

        for k in range(self.num_class):
            tmp = np.exp(-0.5*3/self.beta[k]-0.5*self.nu[k]\
            *(np.dot((self.data-self.m[k]),(self.w.T)[k])*(self.data-self.m[k])).sum(axis=1))
            self.gamma[k] = self.pi[k] * np.sqrt(self.sigma_t[k]) * tmp

        self.gamma /= np.sum(self.gamma, axis=0, keepdims=True)
        self.gamma[np.isnan(self.gamma)] = 1. / self.num_class

    def M_step(self):
        self.alpha = self.alpha0+self.s_k_1
        self.beta = self.beta0+self.s_k_1
        self.m = ((self.beta0*self.m0+self.s_k_x).T/self.beta).T

        for k in range(self.num_class):
            (self.w.T)[k] = np.linalg.inv(np.linalg.inv(self.w0)+self.s_k_xx[k]\
            +self.beta0*np.dot(np.array([self.m0]).T,np.array([self.m0]))\
            -self.beta[k]*np.dot(np.array([self.m[k]]).T,np.array([self.m[k]])))

        self.nu = self.nu0+self.s_k_1

    def calc_likelihood(self):
        z = np.zeros((self.num_class, self.num_data))
        for k in range(self.num_class):
            z[k] = np.array(self.gamma[k]/np.max(self.gamma,axis=0), dtype='int')

        sum = 0
        for k in range(self.num_class):
            sum += np.dot(z[k],np.log(st.multivariate_normal.pdf(self.data, self.m[k], self.sigma_t[k])))
        self.likelihood1 = sum

        sum=0
        self.likelihood2 = np.sum(np.dot(np.log(self.pi),z))

    def vb(self):
        loglikelihood1 = np.zeros(self.ite)
        loglikelihood2 = np.zeros(self.ite)
        for i in range(self.ite):
            print i
            self.E_step()
            self.calc_statistics()
            self.M_step()
            self.calc_likelihood()
            loglikelihood1[i] = self.likelihood1
            loglikelihood2[i] = self.likelihood2
        return self.pi,self.sigma_t,self.gamma, self.alpha,self.beta,self.nu,self.w,self.m,loglikelihood1,loglikelihood2

def main():
    data = input()
    vb = VariationalBayes(4, 100, data)
    pi,sigma_t,gamma,alpha,beta,nu,w,m,loglikelihood1,loglikelihood2 = vb.vb()
    plot3D_class_4(gamma, data)
    plot_loglikelihood(loglikelihood1)
    plot_loglikelihood(loglikelihood2)
    print loglikelihood1
    print loglikelihood2
    output_z(gamma.T)
    output_params(pi,sigma_t,alpha,beta,nu,w.T,m.T)

if __name__ == '__main__':
    main()
