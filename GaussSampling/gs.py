import numpy as np
import csv
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import gamma

def input():
    with open("x.csv") as f:
        reader = csv.reader(f)
        list = [row for row in reader]
    return np.array(list, dtype='float')

def output_z(data):
    with open("z_gs.csv", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in range(len(data)):
            writer.writerow([str(data[i][j]) for j in range(len(data[i]))])

def output_params(pi,mu,sigma,alpha,beta,nu,w,m):
    with open("params_gs.dat", mode='w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Pi'])
        writer.writerow([str(pi[i]) for i in range(len(pi))])
        writer.writerow(['Mu'])
        for i in range(len(mu)):
            writer.writerow([str(mu[i][j]) for j in range(len(mu[i]))])
        writer.writerow(['Sigma'])
        for i in range(len(sigma)):
            writer.writerow(['(class'+str(i+1)+')'])
            for j in range(len(sigma[i])):
                writer.writerow([str(sigma[i][j][k]) for k in range(len(sigma[i][j]))])
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

class GibbsSampling():
    def __init__(self, num_class, ite, data):
        #parameter
        self.pi = np.full(num_class, 1/float(num_class))
        self.mu = 8*np.random.rand(num_class, 3)-4
        self.sigma = np.array([np.diag(np.random.rand(3)+1)]*num_class)
        self.likelihood1 = 0
        self.likelihood2 = 0

        self.z = np.full((num_class, len(data)), 1/float(num_class))
        self.alpha = np.full(num_class,len(data)/num_class)
        self.alpha_0 = np.ones(num_class)
        self.beta = np.full(num_class,len(data)/num_class)
        self.beta_0 = 1.0
        self.nu = np.full(num_class,3)
        self.nu_0 = 3
        self.w = np.tile(np.eye(3), (num_class, 1, 1)).T
        self.w_0 = np.eye(3)
        self.m = 8*np.random.rand(num_class, 3)-4
        self.m_0 = np.zeros(3)

        self.data = data
        self.num_class = num_class
        self.num_data = len(data)
        self.ite = ite

    def sample_z(self):
        deno = np.zeros(self.num_data)
        nume = np.zeros((self.num_class,self.num_data))
        prob = np.zeros((self.num_class,self.num_data))
        for k in range(self.num_class):
            nume[k] = self.pi[k]*st.multivariate_normal.pdf(self.data, self.mu[k], self.sigma[k])
        deno = np.sum(nume, axis=0)

        for k in range(self.num_class):
            prob[k] = nume[k]/deno

        for n in range(self.num_data):
            self.z[:,n] = np.random.multinomial(1,prob[:,n])

    def sample_pi(self):
        s_k_1 = np.sum(self.z, axis = 1)
        self.alpha = self.alpha_0+s_k_1
        self.pi = np.random.dirichlet(self.alpha)

    def sample_mu_sigma(self):
        s_k_1 = np.sum(self.z, axis = 1)
        s_k_x = np.dot(self.z, self.data)
        s_k_xx = np.array([np.diag(np.zeros(3))]*self.num_class)
        for k in range(self.num_class):
            tmp = np.einsum('i,ij->ij', self.z[k], self.data)
            s_k_xx[k] = np.einsum('ij,ik->ijk', tmp, self.data).sum(axis=0)

        self.beta = self.beta_0+s_k_1
        self.m = ((self.beta_0*self.m_0+s_k_x).T/self.beta).T
        self.nu = self.nu_0+s_k_1

        for k in range(self.num_class):
            (self.w.T)[k] = np.linalg.inv(np.linalg.inv(self.w_0)+s_k_xx[k]\
            +self.beta_0*np.dot(np.array([self.m_0]).T,np.array([self.m_0]))\
            -self.beta[k]*np.dot(np.array([self.m[k]]).T,np.array([self.m[k]])))

        for k in range(self.num_class):
            self.mu[k] = np.random.multivariate_normal(self.m[k], np.linalg.inv(self.beta[k]*self.sigma[k]))
            self.sigma[k] = st.wishart.rvs(int(self.nu[k]), self.w.T[k],size = 1)

    def calc_likelihood(self):
        sum = 0
        for k in range(self.num_class):
            sum += np.dot(self.z[k],np.log(st.multivariate_normal.pdf(self.data, self.mu[k], self.sigma[k])))
        self.likelihood1 = sum

        sum=0
        self.likelihood2 = np.sum(np.dot(np.log(self.pi),self.z))

    def gs(self):
        likelihood1 = np.zeros(self.ite)
        likelihood2 = np.zeros(self.ite)
        for i in range(self.ite):
            print i
            self.sample_z()
            self.sample_pi()
            self.sample_mu_sigma()
            self.calc_likelihood()
            likelihood1[i]=self.likelihood1
            likelihood2[i]=self.likelihood2
        return self.pi,self.mu,self.sigma,self.z, self.alpha,self.beta,self.nu,self.w,self.m,likelihood1,likelihood2


def main():
    data = input()
    gs = GibbsSampling(4, 100, data)
    pi,mu,sigma,z,alpha,beta,nu,w,m,likelihood1,likelihood2 = gs.gs()
    plot3D_class_4(z,data)
    plot_loglikelihood(likelihood1)
    plot_loglikelihood(likelihood2)
    print likelihood1
    print likelihood2
    output_z(z.T)
    output_params(pi,mu.T,sigma,alpha,beta,nu,w.T,m.T)


if __name__ == '__main__':
    main()
