import matplotlib.pyplot as plt
import numpy as np

obs_pro=np.array([[1/float(6)]*6,[0.1,0.1,0.1,0.1,0.1,0.5]], dtype=float).T
#hid_pro=np.array([[0.9,0.1],[0.1,0.9]], dtype=float)
hid_pro=np.array([[0.99,0.01],[0.01,0.99]], dtype=float)
obs_data = np.asarray([\
4,4,5,4,2,3,3,6,4,5,5,3,4,4,1,4,5,3,6,5,\
3,3,3,5,5,3,5,6,5,5,1,3,4,3,1,2,6,1,6,1,\
5,4,2,4,1,5,4,1,1,1,1,5,6,6,6,6,1,6,2,6,\
2,6,1,6,6,6,6,6,3,2,6,6,6,1,6,6,2,6,6,5,\
6,6,5,6,6,6,6,4,3,6,6,5,2,5,4,5,6,5,4,4])

def forward():
    alpha = np.zeros(shape=(len(obs_data), 2))
    for t in range(len(obs_data)):
        if(t==0):
            alpha[0]= obs_pro[obs_data[0]-1]
        else:
            alpha[t] = np.dot(hid_pro,alpha[t-1])*obs_pro[obs_data[t]-1]
    return alpha

def backward(init):
    beta = np.zeros(shape=(len(obs_data), 2))
    for t in reversed(range(len(obs_data))):
        if(t==len(obs_data)-1):
            #beta[-1]= obs_pro[obs_data[-1]-1]
            #beta[-1] = np.array([0.3, 0.7])
            beta[-1] = init
        else:
            beta[t] = (obs_pro[obs_data[t+1]-1]*beta[t+1]).dot(hid_pro)
    return beta

def figure(y_1,y_2,y_3):
    x_y1 = [i for i in range(len(y_1))]
    x_y2 = [i for i in range(1,len(y_1))]
    #x_y2 = [i for i in range(len(y_1))]
    x_y3 = [i for i in range(1,len(y_1))]
    #x_y3 = [i for i in range(len(y_1))]

    plt.plot([0,100],[0.5,0.5], color='black', linewidth = 0.5)
    plt.plot(x_y1, y_1, label="alpha", color='blue', linewidth = 0.5)
    plt.plot(x_y2, y_2[:-1], label="beta", color='green', linewidth = 0.5)
    #plt.plot(x_y2, y_2, label="beta", color='green', linewidth = 0.5)
    plt.plot(x_y3, y_3, label="gammma", color='red', linewidth = 0.5)
    for i in xrange(0, len(obs_data), 1):
        plt.annotate(str(obs_data[i]), (i - .75, (5-obs_data[i]) / 30. + 1.0))
    plt.ylim(-0.1, 1.19)
    plt.xlabel("Trial")
    plt.ylabel("Posterior probability")
    plt.legend(loc='lower right')
    plt.show()


def main():
    alpha = forward()
    beta = backward(alpha[-1])
    gamma = np.multiply(alpha[1:],beta[:-1])
    #gamma = np.multiply(alpha,beta)

    alpha /= np.sum(alpha, axis=1, keepdims=True)
    beta /= np.sum(beta, axis=1, keepdims=True)
    gamma /= np.sum(gamma, axis=1, keepdims=True)

    figure(alpha[:,1], beta[:,1], gamma[:,1])

if __name__ == '__main__':
    main()
