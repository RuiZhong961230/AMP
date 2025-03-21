import os
from copy import deepcopy
from opfunu.cec_based.cec2022 import *


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


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])


def CDE(func):
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

        FitOff[i] = func(Off[i])
    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]
            if FitOff[i] < BestFit:
                BestFit = FitOff[i]
                BestPop = deepcopy(Off[i])


def RunCDE(func):
    global curFEs, curIter, MaxIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            CDE(func)
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./DE_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
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
        RunCDE(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./DE_Data/CEC2022') == False:
        os.makedirs('./DE_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
