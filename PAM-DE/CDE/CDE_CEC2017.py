import os
from copy import deepcopy
import numpy as np
from cec17_functions import cec17_test_func


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

BestPop = None
BestFit = float("inf")


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])


def PAM(Off, FitOff, k=10):
    global Pop, FitPop, PopSize
    tmpPop = np.concatenate((Pop, Off))
    tmpFit = np.concatenate((FitPop, FitOff))
    length = len(tmpFit)
    AdjMatrix = np.ones((length, length))
    for i in range(length):  # Construct adjacent matrix
        for j in range(i + 1, length):
            dis = sum(np.abs(tmpPop[i] - tmpPop[j]))
            AdjMatrix[i][j] = max(dis, 0.0001)  # Avoid the distance is 0
            AdjMatrix[j][i] = max(dis, 0.0001)

    Count = np.zeros(length)  # Save winner times
    for i in range(length):
        Dis = deepcopy(AdjMatrix[i])
        Prob = np.zeros(len(Dis))
        for j in range(len(Dis)):  # Closer individual has a higher probability
            Prob[j] = 1 / Dis[j]
        Prob[i] = 0  # Distance with self
        ProbSum = sum(Prob)
        for j in range(len(Dis)):  # Make Dis sum up to 1
            Prob[j] = Prob[j] / ProbSum

        competitors = np.random.choice(list(range(length)), k, replace=False, p=Prob)  # Select competitors with adjacent matrix based probability
        for j in competitors:  # Competition
            if tmpFit[i] < tmpFit[j]:
                Count[i] += 1

    idx_survival = np.argsort(-Count)  # Sort based on winner times
    Pop = deepcopy(tmpPop[idx_survival[0: PopSize]])
    FitPop = tmpFit[idx_survival[0: PopSize]]


def CDE():
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, BestPop, BestFit
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in range(PopSize):
        IDX = np.random.randint(0, PopSize)
        while IDX == i:
            IDX = np.random.randint(0, PopSize)
        candi = list(range(0, PopSize))
        candi.remove(i)
        candi.remove(IDX)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F1 = np.random.normal(0.5, 0.3)
        F2 = np.random.normal(0.5, 0.3)
        if FitPop[IDX] < FitPop[i]:  # DE/winner-to-best/1
            Off[i] = Pop[i] + F1 * (BestPop - Pop[i]) + F2 * (Pop[r1] - Pop[r2])
        else:
            Off[i] = Pop[IDX] + F1 * (BestPop - Pop[IDX]) + F2 * (Pop[r1] - Pop[r2])

        jrand = np.random.randint(0, DimSize)  # bin crossover
        for j in range(DimSize):
            Cr = np.random.normal(0.5, 0.3)
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

        for j in range(DimSize):
            if Off[i][j] < LB[j] or Off[i][j] > UB[j]:
                Off[i][j] = np.random.uniform(LB[j], UB[j])
        FitOff[i] = fitness(Off[i])

    PAM(Off, FitOff)
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])



def RunCDE():
    global curFEs, curIter, MaxIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization()
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            CDE()
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./DE_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim
    FuncNum = 1
    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        RunCDE()


if __name__ == "__main__":
    if os.path.exists('./DE_Data/CEC2017') == False:
        os.makedirs('./DE_Data/CEC2017')
    Dims = [10, 30]
    for Dim in Dims:
        main(Dim)
