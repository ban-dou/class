import numpy as np
import matplotlib.pyplot as plt

#Hyper parameter
Q = np.diag(np.array([1.25,1.0,12.0,0.25]))
R = 0.01
gamma = 0.95
alpha = 0.1
tau = 1/60.0
#Const
C = np.diag(np.array([1/2.4,0.5,180/(12*np.pi),1/1.5]))

class CartPole():
    def __init__(self):
        self.s = np.zeros(4)
        self.x_dd,self.y_dd = 0.0,0.0
        self.sita = 10*np.random.rand(4)-5
        self.eta = 2*np.random.rand()-1
        self.mu = 0.0
        self.sigma = 0.0
        self.a = 0.0
        self.r = 0.0
        self.z = np.zeros(5)
        self.delta = np.array([0.0,0.0,0.0,0.0,1.0])
        self.t = 0.0
        self.lap = np.array([0.0,0.0,0.0,0.0,1.0])
        self.n = 1.0

    def initial(self):
        self.s = np.random.multivariate_normal(np.zeros(4),np.diag(np.full(4, 0.01)))
        self.x_dd, self.y_dd = 0.0,0.0
        self.z = np.zeros(5)
        self.delta = np.zeros(5)
        self.t = 0.0

    def select_action(self):
        self.mu = np.dot(np.dot(self.sita, C), self.s)
        self.sigma = 0.1+1/(1.0+np.exp(self.eta))
        self.a = np.clip(np.random.normal(self.mu, self.sigma),-20.0,20.0)

    def calc_reward(self):
        self.r = -1*np.dot(np.dot(self.s,Q),self.s)-self.a*R*self.a

    def update(self):
        z = self.z
        nabra_sita = (self.a-self.mu)*np.dot(C,self.s)/(self.sigma**2)

        nabra_eta = (self.sigma**2-(self.a-self.mu)**2)*np.exp(self.eta)/\
        (((1+np.exp(self.eta))**2)*np.power(self.sigma,3))

        self.z += np.append(nabra_sita, nabra_eta)
        self.delta += z*self.r*np.power(gamma, self.t)
        self.t += 1.0

    def dynamics(self):
        s = self.s
        x,x_d,y,y_d=self.s[0],self.s[1],self.s[2],self.s[3]
        x_dd,y_dd = self.x_dd, self.y_dd
        M,m,l,g,mu_c,mu_p = 1.0,0.1,0.5,9.8,0.0005,0.000002

        self.s = s+tau*np.array([x_d, x_dd, y_d, y_dd])

        self.y_dd = (g*np.sin(y)+np.cos(y)*(mu_c*np.sign(x_d)-self.a-m*l*(y_d**2)*np.sin(y))\
        /(M+m)-mu_p*y_d/(m*l))/(l*(4/3.0-m*(np.cos(y)**2)/(M+m)))

        self.x_dd = (self.a+m*l*((y_d**2)*np.sin(y)-y_dd*np.cos(y))-mu_c*np.sign(x_d))/(M+m)


    def finish(self):
        flag = 0
        if(np.abs(self.s[0])>2.4):
            flag = 1
        if(np.abs(self.s[1])>2):
            flag = 2
        if(np.abs(self.s[2])>(12*np.pi)/180.0):
            flag = 3
        if(np.abs(self.s[3])>1.5):
            flag = 4
        return flag

    def calc_angel(self, v1, v2):
        inner = np.inner(v1,v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        return np.arccos(np.clip(inner/(v1_norm*v2_norm), -1.0, 1.0))

    def improve(self,i):
        past = self.lap
        self.lap = ((self.n-1.0)*self.lap+self.delta)/self.n
        self.n += 1.0
        if(self.calc_angel(past,self.lap)<0.003 and i != 0):
            self.sita += alpha*self.lap[0:4]
            self.eta += alpha*self.lap[-1]
            self.n = 1.0

    def learn(self,ite):
        count = np.zeros(ite)
        r = np.zeros(ite)
        for i in range(ite):
            self.initial()
            while(1):
                count[i] += 1
                self.select_action()
                self.calc_reward()
                r[i] += self.r*np.power(gamma, self.t)
                self.update()
                self.dynamics()
                if(self.finish()>0):
                    print self.s,self.finish(),self.eta
                    break
            print i,count[i]
            self.improve(i)
        return count, r

def main():
    cp = CartPole()
    count, r = cp.learn(120000)

    plt.xlabel("Iteration number")
    plt.ylabel("Stabilization control time")
    plt.plot(range(len(count)),count)
    plt.show()

    plt.xlabel("Iteration number")
    plt.ylabel("Reward")
    plt.plot(range(len(r)),r)
    plt.show()


if __name__ == '__main__':
    main()
