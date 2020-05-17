import matplotlib.pyplot as plt
import numpy as np

def data_generate(N, gamma, sigma):
    actual = np.zeros(N)
    actual[0] = np.random.normal(0, np.sqrt(gamma))
    for t in range(N-1):
        actual[t+1] = actual[t] + np.random.normal(0, np.sqrt(gamma))
    observe = actual + np.random.normal(0, np.sqrt(sigma), N)
    return actual, observe

def kalman_filter(y, N, gamma, sigma):
    p,k,mu,v = [0.0]*N,[0.0]*N,[0.0]*N,[0.0]*N
    mu[0], v[0] = 0.0, 0.0
    for t in range(1,N):
        if(t<N*2/5 or N*4/5<t):
            p[t-1] = v[t-1]+gamma
            k[t] = p[t-1]/float(p[t-1]+sigma)
            mu[t] = mu[t-1]+k[t]*(y[t]-mu[t-1])
            v[t] = (1-k[t])*p[t-1]
        else:
            p[t-1] = v[t-1]+gamma
            mu[t] = mu[t-1]
            v[t] = v[t-1]+gamma
    return mu, v, p

def main():
    n=500
    gamma = 1.0
    sigma = 49.0

    actual, observe = data_generate(n, gamma, sigma)

    mu, v, p = kalman_filter(observe, n, gamma, sigma)
    x = [i*500/n for i in range(n)]

    plt.errorbar(x, mu, yerr = np.sqrt(v),alpha = 0.1, elinewidth=2, c='steelblue')
    plt.plot(x, mu, c='darkcyan', linewidth=0.5)
    plt.plot(x, actual, c='black', linewidth=0.5)
    plt.scatter(x, observe, c='blue', s=1)
    plt.ylim(min(observe)-1,max(observe)+1)
    plt.show()

if __name__ == '__main__':
    main()
