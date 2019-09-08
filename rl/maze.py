import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import cv2

class Maze():
    def __init__(self,maze,start,goal,eps):
        self.eps = eps
        self.gamma = 0.95
        self.alpha = 0.1
        self.maze = maze
        self.v = np.zeros((len(maze), len(maze[0]))) #Value function
        self.r = 0
        self.p = start #Agent location
        self.p_n = start #Next agent location
        self.s = start #Start location
        self.g = goal #Goal location

    def argmax(self, x, y):
        self.instant_reward(np.array([x,y]))
        return self.r+self.gamma*self.v[x,y]

    def select_action(self):
        tmp = rand()
        cand = np.empty((0,2),int)
        if(tmp>self.eps):
            measure = np.array([])
            if(self.maze[self.p[0]+1,self.p[1]]==1):
                measure = np.append(measure, self.argmax(self.p[0]+1, self.p[1]))
                cand = np.append(cand, np.array([[self.p[0]+1, self.p[1]]]),axis=0)
            if(self.maze[self.p[0]-1,self.p[1]]==1):
                measure = np.append(measure, self.argmax(self.p[0]-1, self.p[1]))
                cand = np.append(cand, np.array([[self.p[0]-1, self.p[1]]]),axis=0)
            if(self.maze[self.p[0],self.p[1]+1]==1):
                measure = np.append(measure, self.argmax(self.p[0], self.p[1]+1))
                cand = np.append(cand, np.array([[self.p[0], self.p[1]+1]]),axis=0)
            if(self.maze[self.p[0],self.p[1]-1]==1):
                measure = np.append(measure, self.argmax(self.p[0], self.p[1]-1))
                cand = np.append(cand, np.array([[self.p[0], self.p[1]-1]]),axis=0)

            if(set(measure)==1):
                index = choice(range(len(cand)))
                self.p_n = cand[index]
            else:
                self.p_n = cand[np.argmax(measure)]
        elif(tmp<self.eps):
            if(self.maze[self.p[0]+1,self.p[1]]==1):
                cand = np.append(cand, np.array([[self.p[0]+1, self.p[1]]]),axis=0)
            if(self.maze[self.p[0]-1,self.p[1]]==1):
                cand = np.append(cand, np.array([[self.p[0]-1, self.p[1]]]),axis=0)
            if(self.maze[self.p[0],self.p[1]+1]==1):
                cand = np.append(cand, np.array([[self.p[0], self.p[1]+1]]),axis=0)
            if(self.maze[self.p[0],self.p[1]-1]==1):
                cand = np.append(cand, np.array([[self.p[0], self.p[1]-1]]),axis=0)
            index = choice(range(len(cand)))
            self.p_n= cand[index]

    def instant_reward(self, vector):
        if(np.array_equal(vector, self.g)):
            self.r = 1
        else:
            self.r = -1

    def update_ef(self):
        x, y = self.p[0], self.p[1]
        x_n, y_n = self.p_n[0], self.p_n[1]
        self.v[x,y]=self.v[x,y]+self.alpha*(self.r+self.gamma*self.v[x_n,y_n]-self.v[x,y])

    def update_ef_q(self):
        x, y = self.p[0], self.p[1]

        measure = np.array([])
        if(self.maze[x+1,y]==1):
            measure = np.append(measure, self.v[x+1, y])

        if(self.maze[x-1,y]==1):
            measure = np.append(measure, self.v[x-1, y])

        if(self.maze[x,y+1]==1):
            measure = np.append(measure, self.v[x, y+1])

        if(self.maze[x,y-1]==1):
            measure = np.append(measure, self.v[x, y-1])

        self.v[x,y]=self.v[x,y]+self.alpha*(self.r+self.gamma*np.max(measure)-self.v[x,y])

    def td(self, ite):
        v = np.zeros(ite)
        p = []
        for i in range(ite):
            print i
            count = 0
            while(True):
                count += 1
                self.select_action()
                self.instant_reward(self.p_n)
                self.update_ef()
                if(np.array_equal(self.p_n,self.g)):
                    p.append(self.p_n)
                    self.p = self.s
                    break
                else:
                    p.append(self.p)
                    self.p = self.p_n
            v[i] = count
        return v, np.array(p)

    def td_q(self, ite):
        v = np.zeros(ite)
        p = []
        for i in range(ite):
            print i
            count = 0
            while(True):
                count += 1
                self.select_action()
                self.instant_reward(self.p_n)
                self.update_ef_q()
                if(np.array_equal(self.p_n,self.g)):
                    p.append(self.p_n)
                    self.p = self.s
                    break
                else:
                    p.append(self.p)
                    self.p = self.p_n
            v[i] = count
        return v, np.array(p)

def make_movie(maze, p):
    image = np.zeros((len(maze), len(maze[0]), 3),dtype='int')
    movie=[]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if(maze[i][j]==1):
                image[i][j]=[255, 255, 255]
            elif(maze[i][j]==0):
                image[i][j]=[0, 0, 0]

    for t in range(len(p)):
        tmp = image.astype(np.uint8)
        tmp[p[t][0]][p[t][1]] =  [0, 165, 255]
        movie.append(tmp)
    return movie

def output(data, width, height, fps, num_frame):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height), True)
    for i in range(num_frame):
        frame = cv2.resize(data[i], (width, height), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        cv2.imshow("Maze search", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyAllWindows()
    return 0

def main():
    '''
    maze =np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,1,1,1,1,1,0],
    [0,1,1,0,1,0,1,1,0,0,1,0],
    [0,0,1,0,1,0,0,0,0,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,0],
    [0,0,0,0,0,1,0,1,0,1,1,0],
    [0,1,1,1,1,1,0,1,0,1,1,0],
    [0,1,0,0,0,0,0,1,0,1,1,0],
    [0,1,1,1,1,0,1,1,0,1,1,0],
    [0,1,1,0,1,1,1,1,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]
    ],dtype='int')
    start = np.array([1,1],dtype='int')
    goal = np.array([10,10],dtype='int')
    '''

    maze =np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0],
    [0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0],
    [0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0],
    [0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0],
    [0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0],
    [0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0],
    [0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0],
    [0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,0],
    [0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0],
    [0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0],
    [0,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0],
    [0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0],
    [0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,0],
    [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0],
    [0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0],
    [0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0],
    [0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0],
    [0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
    [0,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0],
    [0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],dtype='int')
    start = np.array([1,1],dtype='int')
    goal = np.array([23,23],dtype='int')

    eps=0.0
    m = Maze(maze, start, goal, eps)
    v,p = m.td(1500)
    m_q = Maze(maze, start, goal, eps)
    v_q,p_q = m_q.td_q(1500)
    plt.plot(range(len(v)), v, label='SARSA', alpha=0.5)
    plt.plot(range(len(v_q)), v_q, label='Q-learning', alpha=0.5)

    '''
    eps = [0.0,0.1,0.5]
    for i in range(len(eps)):
        m = Maze(maze, start, goal, eps[i])
        #v,p = m.td(1500)
        v,p = m.td_q(1500)
        plt.plot(range(len(v)), v, label=str(eps[i]), alpha=0.7)
    '''
    '''
    data = make_movie(maze, p)
    width, height, fps, num_frame = 396, 396, 30, len(p)
    output(data, width, height, fps, num_frame)
    '''

    plt.xlabel("Iteration number")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
