import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def input(num_frame):
    movie = []
    for i in range(num_frame):
        image = []
        f = open('txt/img'+str(i+1).zfill(3)+'.txt')
        lines = f.readlines()
        for line in lines:
            tmp = []
            for j in range(40):
                if(int(line[j])==0):
                    tmp.append([0,0,0])
                elif(int(line[j])==1):
                    tmp.append([255,255,255])
            image.append(tmp)
        f.close()
        movie.append(np.array(image).astype(np.uint8))
    return movie


def output(data, width, height, fps, num_frame):
    width, height, fps, num_frame = 436, 344, 30, 400
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height), True)
    for i in range(num_frame):
        frame = cv2.resize(data[i], (width, height), interpolation=cv2.INTER_AREA)

        if(i%5==0):
            cv2.imwrite('figfig'+str(i)+'.png',frame)

        writer.write(frame)
        cv2.imshow("Particle filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyAllWindows()
    return 0

class ParticleFilter():
    def __init__(self, width, height, num_frame):
        self.width = width
        self.height = height
        self.num_frame = num_frame
        self.pix_wid = 40
        self.pix_hei = 30
        self.part_wid = 6
        self.part_hei = 8
        self.num_part = self.part_wid*self.part_hei
        self.n = 7
        self.sigma = 1
        self.x = np.zeros((num_frame, self.num_part, 2), dtype='int')
        self.w = np.zeros((num_frame, self.num_part))
        self.w_update = 5
        self.expect = np.zeros((num_frame, 2), dtype='int')
        self.back_pro = 1/4.0

    '''
    def sampling(self, t):
        if(t==0):
            tmp = []
            for i in range(0,self.part_hei):
                for j in range(0,self.part_wid):
                    x = (self.n-1)/2+(i*(self.pix_hei-self.n))/(self.part_hei-1)
                    y = (self.n-1)/2+(j*(self.pix_wid-self.n))/(self.part_wid-1)
                    tmp.append([x ,y])
            self.x[t] = np.array(tmp, dtype='int')
        else:

            eps = np.array(np.random.normal(0, np.sqrt(self.sigma),(self.num_part, 2)),dtype='int')
            self.x[t] = self.x[t-1]+eps

            for i in range(self.num_part):
                if(self.x[t][i][0] < (self.n-1)/2):
                    self.x[t][i][0]=(self.n-1)/2
                elif(self.x[t][i][0]>(self.n-1)/2+(self.pix_hei-self.n)):
                    self.x[t][i][0]=(self.n-1)/2+(self.pix_hei-self.n)

                if(self.x[t][i][1]<(self.n-1)/2):
                    self.x[t][i][1]=(self.n-1)/2
                elif(self.x[t][i][1]>(self.n-1)/2+(self.pix_wid-self.n)):
                    self.x[t][i][1]=(self.n-1)/2+(self.pix_wid-self.n)
    '''

    def sampling(self, t):
        if(t==0):
            tmp = []
            for i in range(0,self.part_hei):
                for j in range(0,self.part_wid):
                    x = (self.n-1)/2+(i*(self.pix_hei-self.n))/(self.part_hei-1)
                    y = (self.n-1)/2+(j*(self.pix_wid-self.n))/(self.part_wid-1)
                    tmp.append([x ,y])
            self.x[t] = np.array(tmp, dtype='int')
        else:
            eps = np.array(np.random.normal(0, np.sqrt(self.sigma),(self.num_part, 2)),dtype='int')
            if(t==1):
                self.x[t] = self.x[t-1]+eps
            else:
                c=1
                self.x[t] = self.x[t-1]+c*(self.x[t-1]-self.x[t-2])+eps


            for i in range(self.num_part):
                if(self.x[t][i][0] < (self.n-1)/2):
                    self.x[t][i][0]=(self.n-1)/2
                elif(self.x[t][i][0]>(self.n-1)/2+(self.pix_hei-self.n)):
                    self.x[t][i][0]=(self.n-1)/2+(self.pix_hei-self.n)

                if(self.x[t][i][1]<(self.n-1)/2):
                    self.x[t][i][1]=(self.n-1)/2
                elif(self.x[t][i][1]>(self.n-1)/2+(self.pix_wid-self.n)):
                    self.x[t][i][1]=(self.n-1)/2+(self.pix_wid-self.n)

    '''
    def reweightning(self, observe, t):
        a = []
        for i in range(self.num_part):
            count = 0
            for j in range(-(self.n-1)/2, 1+(self.n-1)/2):
                for k in range(-(self.n-1)/2, 1+(self.n-1)/2):
                    tmp = observe[self.x[t][i][0]+j][self.x[t][i][1]+k]
                    if(tmp[0]==255):
                        count += 1
            a.append(count)

        if(t==0):
            for i in range(self.num_part):
                self.w[t][i] = 1/float(self.num_part)
        else:
            for i in range(self.num_part):
                comb = math.factorial(self.n*self.n)//(math.factorial(self.n*self.n-a[i])*math.factorial(a[i]))
                self.w[t][i] = self.w[t-1][i]*comb*pow(3/4.0, a[i])*pow(1/4.0, self.n*self.n-a[i])

            self.w[t] = self.w[t]/np.sum(self.w[t])
    '''

    def reweightning(self, observe, t):
        a = []
        b = []
        num_out = []
        flag = 0
        for i in range(self.num_part):
            count = 0
            for j in range(-(self.n-1)/2, 1+(self.n-1)/2):
                for k in range(-(self.n-1)/2, 1+(self.n-1)/2):
                    tmp = observe[self.x[t][i][0]+j][self.x[t][i][1]+k]
                    if(tmp[0]==255):
                        count += 1
            a.append(count)

        for i in range(self.num_part):
            c_b = 0
            c_n = 0
            n_out = self.n+4

            for j in range(-(n_out-1)/2, 1+(n_out-1)/2):
                for k in range(-(n_out-1)/2, 1+(n_out-1)/2):
                    if((self.x[t][i][0]+j)<self.pix_hei and (self.x[t][i][1]+k)<self.pix_wid):
                        c_n += 1
                        tmp = observe[self.x[t][i][0]+j][self.x[t][i][1]+k]
                        if(tmp[0]==255):
                            c_b += 1
            b.append(c_b-a[i])
            num_out.append(c_n-self.n*self.n)

        if(t==0):
            for i in range(self.num_part):
                self.w[t][i] = 1/float(self.num_part)
        else:
            if(np.average(np.array(b)/np.array(num_out,dtype='float'))>0.9 and t<100):
                self.back_pro = 1/3.0
            elif(np.average(np.array(b)/np.array(num_out,dtype='float'))>0.9 and 100<t<200):
                self.back_pro = 5/12.0
            elif(np.average(np.array(b)/np.array(num_out,dtype='float'))>0.9 and 200<t<300):
                self.back_pro = 1/2.0

            for i in range(self.num_part):
                comb = math.factorial(self.n*self.n)//(math.factorial(self.n*self.n-a[i])*math.factorial(a[i]))
                p_1 = comb*pow(3/4.0, a[i])*pow(1/4.0, self.n*self.n-a[i])

                comb = math.factorial(num_out[i])//(math.factorial(num_out[i]-b[i])*math.factorial(b[i]))
                p_2 = comb*pow(self.back_pro, b[i])*pow(1.0-self.back_pro, num_out[i]-b[i])

                self.w[t][i] = self.w[t-1][i]*p_1*p_2

            self.w[t] = self.w[t]/np.sum(self.w[t])

    def resampling(self, t):
        if(t!=0 or t!=self.num_frame-1):
            tmp = self.w[t].copy()
            index = np.argsort(tmp)
            for i in range(self.w_update):
                self.x[t][index[i]][0] = self.x[t][index[-1]][0]
                self.x[t][index[i]][1] = self.x[t][index[-1]][1]
                self.w[t][index[i]] = self.w[t][index[-1]]

    def average_weighted(self, t):
        tmp = np.dot(self.w[t].T, self.x[t])
        self.expect[t][0] = tmp[0]
        self.expect[t][1] = tmp[1]

    def plot_object(self, data):
        for t in range(self.num_frame):
            for i in range(-(self.n-1)/2, 1+(self.n-1)/2):
                if(abs(i)==(self.n-1)/2):
                    for j in range(-(self.n-1)/2, 1+(self.n-1)/2):
                        data[t][self.expect[t][0]+i][self.expect[t][1]+j] = [0, 165, 255]
                else:
                    data[t][self.expect[t][0]+i][self.expect[t][1]-(self.n-1)/2] = [0, 165, 255]
                    data[t][self.expect[t][0]+i][self.expect[t][1]+(self.n-1)/2] = [0, 165, 255]
        return data

    def filter(self, data):
        for t in range(self.num_frame):
            print t
            self.sampling(t)
            self.reweightning(data[t], t)
            self.average_weighted(t)
            self.resampling(t)
        data = self.plot_object(data)
        return data

def main():
    width, height, fps, num_frame = 436, 344, 30, 400
    data = input(num_frame)

    hmm = ParticleFilter(width, height, num_frame)
    data = hmm.filter(data)

    output(data, width, height, fps, num_frame)

if __name__ == '__main__':
    main()
