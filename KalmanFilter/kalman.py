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

def kalman_smoother(y, N, gamma, mu, v, p):
    j,mu_h,v_h = [0.0]*N,[0.0]*N,[0.0]*N
    mu_h[-1], v_h[-1] = mu[-1], v[-1]
    for t in reversed(range(len(y)-1)):
        j[t] = v[t]/p[t]
        mu_h[t] = mu[t] + j[t]*(mu_h[t+1]-mu[t])
        v_h[t] = v[t] + j[t]*(v_h[t+1]-p[t])*j[t]
    return mu_h,v_h

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


    mu_h, v_h = kalman_smoother(observe, n, gamma, mu, v, p)
    plt.errorbar(x, mu_h, yerr = np.sqrt(v_h),alpha = 0.3, elinewidth=2, c='red')
    plt.plot(x, mu_h, c='green', linewidth=0.5)
    plt.plot(x, actual, c='black', linewidth=0.5)
    plt.scatter(x, observe, c='blue', s=1)
    plt.show()

if __name__ == '__main__':
    main()
