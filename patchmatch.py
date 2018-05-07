import cv2
import numpy as np
import time
import rawpy
import matplotlib.pyplot as plt
import os

class NNF:
    def __init__(self, img_a, img_b, boxsize=30):
    	
        self.A = img_a
        self.B = img_b
        self.aa = np.zeros((self.A.shape[0], self.A.shape[1], 3)).astype(np.int)
        self.bb = np.zeros((self.B.shape[0], self.B.shape[1], 3)).astype(np.int)
        for i in range(self.aa.shape[0]):
            for j in range(self.bb.shape[1]):
                for k in range(3):
                    self.aa[i,j,k] = self.A[i,j,k]
                    self.bb[i,j,k] = self.B[i,j,k]
        self.boxsize = boxsize
        self.flag = 0
        #return correspondences in three channels: x_coordinates, y_coordinates, offsets
        self.nnf = np.zeros((2, self.A.shape[0], self.A.shape[1])).astype(np.int)
        self.nnf_D = np.zeros((self.A.shape[0], self.A.shape[1]))
        self.init_nnf()

    def init_nnf(self):
        # self.nnf[0] = np.random.randint(self.B.shape[0], size=(self.A.shape[0], self.A.shape[1]))
        # self.nnf[1] = np.random.randint(self.B.shape[1], size=(self.A.shape[0], self.A.shape[1]))
        print("start init")
        self.nnf = self.nnf.transpose((1, 2 ,0)) 
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                self.nnf[i, j] = [i, j]
                # pos = self.nnf[i,j]
                self.nnf_D[i,j] = self.cal_dist(i, j, i, j)
        print("finish init")


    def cal_dist(self, ai ,aj, bi, bj):
        # check boundary
        dx = dy = self.boxsize
        dx = min(self.A.shape[0]-ai, dx)
        dy = min(self.A.shape[1]-aj, dy)
        if bi + dx > self.B.shape[0]:
            bi = self.B.shape[0] - dx
        if bj + dy > self.B.shape[1]:
            bj = self.B.shape[1] - dy
        return np.sum((self.aa[ai:ai+dx, aj:aj+dy]-self.bb[bi:bi+dx, bj:bj+dy])**2) / dx / dy
        #self.nnf_D[i,j] =np.sum((self.A[i-dx0:i+dx1, j-dy0:j+dy1]-self.B[i-dx0:i+dx1, j-dy0:j+dy1])**2)
    #def improve_guess(self,):


    def improve_nnf(self, total_iter=5):
        for iter in range(1, total_iter+1):    
        	
            if iter % 2 > 0:   # odd
            	# print("odd")
                Irange = range(0, self.A.shape[0])
                Jrange = range(0, self.A.shape[1])
                d = -1
            else:   # even
            	# print("even")
                Irange = np.arange(self.A.shape[0] - 1, -1, -1)
                Jrange = np.arange(self.A.shape[1] - 1, -1, -1)
                d = 1
            print("iteration: %d" % iter)
            self.flag = 0
            print("Temp Flag: %d" % self.flag)
            for i in Irange:
                for j in Jrange:
                    pos = self.nnf[i,j]
                    x, y = pos[0], pos[1]
                    bestx, besty, bestd = x, y, self.nnf_D[i,j]

                    if i < 0 or j < 0 or i >= self.A.shape[0] or j >= self.A.shape[1]:
                    	print("wrong")
                    	pass
                    # if i < 5 and j < 5:
                    # 	print(i,j)
                    # 	pass

                    if d < 0:  # odd   d=-1
                        if i+d >= 0:
                            rx, ry = self.nnf[i+d, j][0], self.nnf[i+d, j][1]
                            if rx < 0 or ry < 0:
                            	print("d: %d; i: %d" % (d, i))
                            	pass
                            if rx < self.B.shape[0]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    if i == 200 and j == 300:
                                        print('bestx: %d, besty: %d, bestd: %d; rx: %d, ry: %d, val: %d' % (bestx, besty, bestd, rx, ry, val))
                                    bestx, besty, bestd = rx, ry, val
                                    self.flag = 1


                        if j+d >= 0:
                            rx, ry = self.nnf[i, j+d][0], self.nnf[i, j+d][1]
                            if rx < 0 or ry < 0:
                            	print("d: %d; j: %d" % (d, j))
                            	pass
                            if ry < self.B.shape[1]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    if i == 200 and j == 300:
                                        print('bestx: %d, besty: %d, bestd: %d; rx: %d, ry: %d, val: %d' % (bestx, besty, bestd, rx, ry, val))
                                    bestx, besty, bestd = rx, ry, val
                                    self.flag = 1
                    else:    # even d=1
                        if i+d < self.A.shape[0]:
                            rx, ry = self.nnf[i+d, j][0], self.nnf[i+d, j][1]
                            if rx < 0 or ry < 0:
                            	print("d: %d; i: %d" % (d, i))
                            if rx >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    if i == 200 and j == 300:
                                        print('bestx: %d, besty: %d, bestd: %d; rx: %d, ry: %d, val: %d' % (bestx, besty, bestd, rx, ry, val))
                                    bestx, besty, bestd = rx, ry, val
                                    self.flag = 1

                        if j+d < self.A.shape[1]:
                            rx, ry = self.nnf[i, j+d][0], self.nnf[i, j+d][1]
                            if rx < 0 or ry < 0:
                            	print("d: %d; j: %d" % (d, j))
                            if ry >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    if i == 200 and j == 300:
                                        print('bestx: %d, besty: %d, bestd: %d; rx: %d, ry: %d, val: %d' % (bestx, besty, bestd, rx, ry, val))
                                    bestx, besty, bestd = rx, ry, val
                                    self.flag = 1


                    # rand_d = min(self.B.shape[0]//2, self.B.shape[1]//2)
                    rand_d = max(self.B.shape[0], self.B.shape[1])
                    while rand_d > 0:
                        try:
                            xmin = max(bestx - rand_d, 0)
                            xmax = min(bestx + rand_d, self.B.shape[0])
                            ymin = max(besty - rand_d, 0)
                            ymax = min(besty + rand_d, self.B.shape[1])

                            rx = np.random.randint(xmin, xmax)
                            ry = np.random.randint(ymin, ymax)


                            val = self.cal_dist(i, j, rx, ry)
                            if val < bestd:
                                if i == 200 and j == 300:
                                    print('bestx: %d, besty: %d, bestd: %d; rx: %d, ry: %d, val: %d' % (bestx, besty, bestd, rx, ry, val))
                                
                                bestx, besty, bestd = rx, ry, val
                                self.flag = 1
                        except:
                            print("rand_d: %d" % rand_d)
                            print("xmin: %d; xmax: %d" % (xmin, xmax))
                            print("ymin: %d; ymax: %d" % (ymin, ymax))
                            print("bestx: %d; besty: %d" % (bestx, besty))
                            print(self.B.shape)
                      #   if i < 2 and j < 2:
                    		# print(rand_d)
                    		# pass
                        rand_d = rand_d // 2
                    # if i == 130 and j == 240:
                    #     print("x: %d; y: %d; newx: %d; newy: %d" % (x, y, bestx, besty))
                    self.nnf[i, j] = [bestx, besty]
                    self.nnf_D[i, j] = bestd


    def solve(self):
        print("boxsize: %d" % (self.boxsize))
        self.improve_nnf(total_iter=5) 
        print("Flag: %d" % self.flag)

    def reconstruct(self):
        ans = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = self.nnf[i,j]
                ans[i,j] = self.B[pos[0], pos[1]]
        return ans

    def demo(self, x, y):
        pos = self.nnf[x, y]
        rx, ry = pos[0], pos[1]
        dx = dy = self.boxsize
        dx = min(dx, self.A.shape[0]-x) 
        dy = min(self.A.shape[1]-y, dy)
        if rx + dx > self.B.shape[0]:
            rx = self.B.shape[0] - dx
        if ry + dy > self.B.shape[1]:
            ry = self.B.shape[1] - dy
        retA = np.tile(self.A, 1)
        retB = np.tile(self.B, 1)
        for i in range(x,x+dx):
            retA[i][y] = [255, 0, 0]
        for i in range(x,x+dx):
            retA[i][y+dy-1] = [255, 0, 0]	
        for j in range(y, y+dy):
            retA[x][j] = [255, 0, 0]
        for j in range(y, y+dy):
            retA[x+dx-1][j] = [255, 0, 0]
        for i in range(rx,rx+dx):
            retB[i][ry] = [255, 0, 0]
        for i in range(rx,rx+dx):
            retB[i][ry+dy-1] = [255, 0, 0]
        for j in range(ry, ry+dy):
            retB[rx][j] = [255, 0, 0]
        for j in range(ry, ry+dy):
            retB[rx+dx-1][j] = [255, 0, 0]
        return retA, retB


    def primeReconstruct(self):
        ans = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                dx = dy = self.boxsize
                pos = self.nnf[i,j]
                dx = min(dx, self.A.shape[0]-i, self.B.shape[0]-pos[0]) 
                dy = min(self.A.shape[1]-j, dy, self.B.shape[1]-pos[1])
                bestx, besty = self.findNearestPxl(i, j, pos[0], pos[1], dx, dy)
                ans[i,j] = self.B[bestx, besty]
        return ans


    def findNearestPxl(self, x, y, rx, ry, dx, dy): 
        bestx = rx
        besty = ry
        bestd = np.sum((self.aa[x, y] - self.bb[rx, ry])**2)
        for i in range(rx, rx + dx):
            for j in range(ry, ry + dy):
                val = np.sum((self.aa[x, y] - self.bb[i, j])**2)
                if val < bestd:
                    bestx = i
                    besty = j
                    bestd = val
        return bestx, besty





    # a = cv2.imread('project_a.png')
    # b = cv2.imread('project_b.png')
    # start = time.time()
    # print('started')
    # nnf = NNF(a, b, boxsize=30) 
    # nnf.solve()
    # end = time.time()
    # print(end - start)
    # print('reconstruct image')
    # # c = nnf.reconstruct()
    # c = nnf.primeReconstruct()
    # cv2.imwrite('project_30.png', c)
    # print('finish reconstruct')
    
  