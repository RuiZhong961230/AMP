from copy import deepcopy
import os
from opfunu.cec_based.cec2022 import *
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


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def Initialization(func):
    global Pop, FitPop, curFEs, DimSize
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


def JADE(func):
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

        FitOff[i] = func(Off[i])
        if FitOff[i] < FitPop[i]:
            F_List.append(F)
            Cr_List.append(Cr)

    PAM(Off, FitOff)

    if len(F_List) == 0:
        pass
    else:
        muF = (1 - c) * muF + c * meanL(F_List)
    if len(Cr_List) == 0:
        pass
    else:
        muCr = (1 - c) * muCr + c * np.mean(Cr_List)


def RunJADE(func):
    global curFEs, curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize, muF, muCr
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        muF, muCr = 0.5, 0.5
        Initialization(func)
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            JADE(func)
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./DE_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(Dim):
    global FuncNum, DimSize, Pop, LB, UB, MaxIter, MaxFEs

    DimSize = Dim
    LB, UB = [-100] * Dim, [100] * Dim
    MaxFEs = 1000 * DimSize
    MaxIter = int(MaxFEs / PopSize)
    Pop = np.zeros((PopSize, DimSize))

    CEC2022 = [F12022(DimSize), F22022(DimSize), F32022(DimSize), F42022(DimSize), F52022(DimSize), F62022(DimSize),
               F72022(DimSize), F82022(DimSize), F92022(DimSize), F102022(DimSize), F112022(DimSize), F122022(DimSize)]

    for i in range(len(CEC2022)):
        FuncNum = i + 1
        RunJADE(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('DE_Data/CEC2022') == False:
        os.makedirs('DE_Data/CEC2022')
    for dim in [10, 20]:
        main(dim)
