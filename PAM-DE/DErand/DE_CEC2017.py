import os
import numpy as np
from copy import deepcopy
from cec17_functions import cec17_test_func


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
FuncNum = 1
MaxIter = int(MaxFEs / PopSize)


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


def Initialization():
    global Pop, FitPop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])


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


def DE():
    global Pop, FitPop
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in range(PopSize):  # Mutation
        r1, r2, r3 = np.random.choice(list(range(PopSize)), 3, replace=False)
        Off[i] = Pop[r1] + 0.8 * (Pop[r2] - Pop[r3])
        Off[i] = np.clip(Off[i], LB, UB)

    for i in range(PopSize):  # Crossover
        jrand = np.random.randint(0, DimSize)
        for j in range(DimSize):
            if np.random.rand() < 0.7 or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

        FitOff[i] = fitness(Off[i])
    PAM(Off, FitOff)  # PAM Selection


def RunDE():
    global Pop, FitPop, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        Initialization()
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            DE()
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./DE_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunDE()


if __name__ == "__main__":
    if os.path.exists('DE_Data/CEC2017') == False:
        os.makedirs('DE_Data/CEC2017')
    Dims = [10, 30]
    for Dim in Dims:
        main(Dim)


