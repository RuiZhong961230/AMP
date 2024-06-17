from copy import deepcopy
import os
from cec17_functions import cec17_test_func
import numpy as np
from scipy.stats import cauchy


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize
curFEs = 0
MaxIter = int(MaxFEs / PopSize)
curIter = 0
Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

muF = 0.5
muCr = 0.5


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def Initialization():
    global Pop, FitPop, curFEs, DimSize
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])


def JADE():
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, muF, muCr
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    F_List, Cr_List = [], []
    idx_sort = np.argsort(FitPop)
    Xpbest = np.mean(Pop[idx_sort[0: int(0.05 * PopSize)]], axis=0)
    c, sigma = 0.1, 0.1
    for i in range(PopSize):
        candi = list(range(0, PopSize))
        candi.remove(i)
        r2, r3 = np.random.choice(candi, 2, replace=False)
        F = cauchy.rvs(muF, sigma)
        while True:
            if F > 1:
                F = 1
                break
            elif F < 0:
                F = cauchy.rvs(muF, sigma)
            break
        Off[i] = Pop[i] + F * (Xpbest - Pop[i]) + F * (Pop[r2] - Pop[r3])
        Off[i] = np.clip(Off[i], LB, UB)

        Cr = np.clip(np.random.normal(muCr, sigma), 0, 1)
        jrand = np.random.randint(0, DimSize)
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

        FitOff[i] = fitness(Off[i])
        if FitOff[i] < FitPop[i]:
            F_List.append(F)
            Cr_List.append(Cr)

    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]

    if len(F_List) == 0:
        pass
    else:
        muF = (1 - c) * muF + c * meanL(F_List)
    if len(Cr_List) == 0:
        pass
    else:
        muCr = (1 - c) * muCr + c * np.mean(Cr_List)


def RunJADE():
    global curFEs, curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize, muF, muCr
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        muF, muCr = 0.5, 0.5
        Initialization()
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            JADE()
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./DE_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(Dim):
    global FuncNum, DimSize, Pop, LB, UB, MaxIter, MaxFEs

    DimSize = Dim
    LB, UB = [-100] * Dim, [100] * Dim
    MaxFEs = 1000 * DimSize
    MaxIter = int(MaxFEs / PopSize)
    Pop = np.zeros((PopSize, DimSize))

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunJADE()


if __name__ == "__main__":
    if os.path.exists('DE_Data/CEC2017') == False:
        os.makedirs('DE_Data/CEC2017')
    Dims = [10, 30]
    for dim in Dims:
        main(dim)
