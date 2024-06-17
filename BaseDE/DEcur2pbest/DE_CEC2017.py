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


def DE():
    global Pop, FitPop
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    idx_sort = np.argsort(FitPop)
    Xpbest = np.mean(Pop[idx_sort[0: int(0.05 * PopSize)]], axis=0)
    for i in range(PopSize):  # Mutation
        candi = list(range(PopSize))
        candi.remove(i)
        r2, r3 = np.random.choice(candi, 2, replace=False)
        Off[i] = Pop[i] + 0.8 * (Xpbest - Pop[i]) + 0.8 * (Pop[r2] - Pop[r3])
        Off[i] = np.clip(Off[i], LB, UB)

    for i in range(PopSize):  # Crossover
        jrand = np.random.randint(0, DimSize)
        for j in range(DimSize):
            if np.random.rand() < 0.7 or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

    for i in range(PopSize):  # Selection
        FitOff[i] = fitness(Off[i])
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]


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


