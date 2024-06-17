import os
from copy import deepcopy
from opfunu.cec_based.cec2022 import *


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


def Initialization(func):
    global Pop, FitPop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])


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


def DE(func):
    global Pop, FitPop
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in range(PopSize):  # Mutation
        candi = list(range(PopSize))
        candi.remove(i)
        r2, r3 = np.random.choice(candi, 2, replace=False)
        Off[i] = Pop[i] + 0.8 * (Pop[r2] - Pop[r3])
        Off[i] = np.clip(Off[i], LB, UB)

    for i in range(PopSize):  # Crossover
        jrand = np.random.randint(0, DimSize)
        for j in range(DimSize):
            if np.random.rand() < 0.7 or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]
        FitOff[i] = func(Off[i])

    PAM(Off, FitOff)  # PAM Selection


def RunDE(func):
    global Pop, FitPop, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        Initialization(func)
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            DE(func)
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./DE_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(DimSize), F22022(DimSize), F32022(DimSize), F42022(DimSize), F52022(DimSize), F62022(DimSize),
               F72022(DimSize), F82022(DimSize), F92022(DimSize), F102022(DimSize), F112022(DimSize), F122022(DimSize)]

    for i in range(len(CEC2022)):
        FuncNum = i + 1
        RunDE(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('DE_Data/CEC2022') == False:
        os.makedirs('DE_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)


