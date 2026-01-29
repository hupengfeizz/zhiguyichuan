import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警
# 设置显示中文字体，
import matplotlib.pyplot as plt
from pylab import mpl
# 设置font.sans-serif 或 font.family 均可
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"] ## mac
plt.rcParams['font.family']=['Arial Unicode MS'] ## mac
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
import numpy as np
from sys import stdout
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import copy
from sklearn.model_selection import train_test_split
import scipy.io as scio
import pandas as pd
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1,scale=False)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc,scale=False)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

def CARS_Cloud(X, y, N=100, f=10, cv=5):
    p = 0.8
    m, n = X.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    # val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    Coeff = []
    WaveNum =[]
    RMSECV = []
    r = []
    band_list = []
    comp = []
    for i in range(1, N+1):
        print(i,x.shape)
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        #WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice    \
            (np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        #xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1,-1)
        wnum = n - wave_num
        print(wnum)
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum),-2)))

        if len(WaveData) == 0:
            WaveData = d
        else:
            print(WaveData.shape,d.shape)
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f,scale=False)
        pls.fit(xcal, ycal)
        beta = pls.coef_

        coef = copy.deepcopy(beta)
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)


        coeff = coef[b2, :].reshape(len(b2), -1)
        cb = coeff[:wave_num]

        if wnum > 0:
            cb = np.vstack((cb, np.full((wnum, 1), -2)))
        if len(Coeff) == 0:
            Coeff = copy.deepcopy(cb)
        else:
            print(Coeff.shape,cb.shape)
            Coeff = np.hstack((Coeff, cb))

        # if xcal.shape[1]>3:
        #     if xcal.shape[1]-3<f:
        #         f = xcal.shape[1]-3
        #     bands, opt_comp, rms = band_selection_sa(xcal, ycal, n_of_bands=xcal.shape[1]-3, max_lv=f, n_iter=20)
        #     print(rms)
        #     RMSECV.append(rms)
        #     band_list.append(bands)
        #     comp.append(opt_comp)
        # else:
        #     rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        #     band_list.append(xcal)
        #     comp.append(rindex+1)
        #     rms = Cross_Validation(xcal, ycal, rindex+1, cv)
        #     RMSECV.append(rms)
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        rms = Cross_Validation(xcal, ycal, rindex + 1, cv)
        RMSECV.append(rms)
    CoeffData = Coeff.T

    WAVE = []
    COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        cd = CoeffData[i, :]
        WD = np.zeros((len(wd)))
        CO = np.zeros((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                CO[j] = 0
            else:
                print(ind)
                WD[j] = wd[ind[0]]
                CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        if len(COEFF) == 0:
            COEFF = copy.deepcopy(CO)
        else:
            COEFF = np.vstack((COEFF, CO.reshape(1, -1)))
    #print(WaveData)
    #print(WAVE)
    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]
    OptWave=np.array(OptWave)
    OptWave=list(OptWave)
    print(COEFF.shape)
    # fig = plt.figure()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.figure(figsize=(8, 6))
    # 设置刻度的间隔为每隔10一个刻度
    x_ticks = np.arange(0, N + 10, 10)
    plt.xticks(x_ticks,fontsize=16)
    plt.yticks(fontsize=16)
    # plt.subplot(211)
    # plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    # plt.ylabel('The number of wavelengths selected', fontsize=fonts)
    # plt.title('The optimal number of iterations:' + str(MinIndex) + 'times', fontsize=fonts)
    # plt.plot(np.arange(N), WaveNum)

    # plt.subplot(212)
    # plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    # plt.ylabel('RMSECV', fontsize=fonts)
    # plt.plot(np.arange(N), RMSECV)

    plt.xlabel('(c) Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    for  i in range(COEFF.shape[1]):

        plt.plot(np.arange(COEFF.shape[0]),COEFF[:,i])
    plt.vlines(MinIndex, -1e4, 1e4, colors='r')
    # plt.savefig(r'/Users/macbook/Documents/data/FL200_N.xlsx')
    plt.show()
    return (OptWave)

def pls_optimise_components(X, y, npc):

    rmsecv = np.zeros(npc)
    for i in range(1,npc+1,1):

        # Simple PLS
        pls_simple = PLSRegression(n_components=i)
        # Fit
        pls_simple.fit(X, y)
        # Cross-validation
        y_cv = cross_val_predict(pls_simple, X, y, cv=10)

        # Calculate scores
        score = r2_score(y, y_cv)
        rmsecv[i-1] = np.sqrt(mean_squared_error(y, y_cv))

    # Find the minimum of ther RMSE and its location
    opt_comp, rmsecv_min = np.argmin(rmsecv),  rmsecv[np.argmin(rmsecv)]

    return (opt_comp+1, rmsecv_min)

def band_selection_sa(X,y,n_of_bands, max_lv, n_iter):

    p = np.arange(X.shape[1])
    np.random.shuffle(p)
    bands = p[:n_of_bands] # Selected Bands. Start off with a random selection
    nbands = p[n_of_bands:] # Excluded bands

    Xop = X[:,bands] #This is the array to be optimised

    # Run a PLS optimising the number of latent variables
    opt_comp, rmsecv_min = pls_optimise_components(Xop, y, max_lv)

    rms = [] # Here we store the RMSE value as the optimisation progresses
    for i in range(n_iter):

        cool = 0.001*rmsecv_min # cooling parameter. It decreases with the RMSE
        new_bands = np.copy(bands)
        new_nbands = np.copy(nbands)

        # swap three elements at random
        for jj in range(3):
            r1, r2 = np.random.randint(n_of_bands),np.random.randint(X.shape[1]-n_of_bands)
            el1, el2 = new_bands[r1],new_nbands[r2]
            new_bands[r1] = el2
            new_nbands[r2] = el1


        Xop = X[:,new_bands]

        opt_comp_new, rmsecv_min_new = pls_optimise_components(Xop, y, max_lv)

        # If the new RMSE is less than the previous, accept the change
        if (rmsecv_min_new < rmsecv_min):
            bands = new_bands
            nbands = new_nbands
            opt_comp = opt_comp_new
            rmsecv_min = rmsecv_min_new
            rms.append(rmsecv_min_new)

            stdout.write("\r"+str(i))
            stdout.write(" ")
            stdout.write(str(opt_comp_new))
            stdout.write(" ")
            stdout.write(str(rmsecv_min_new))
            stdout.flush()

        # If the new RMSE is larger than the previous, accept it with some probability
        # dictated by the cooling parameter
        if (rmsecv_min_new > rmsecv_min):

            prob = np.exp(-(rmsecv_min_new - rmsecv_min)/cool) # probability
            if (np.random.random() < prob):
                bands = new_bands
                nbands = new_nbands
                opt_comp = opt_comp_new
                rmsecv_min = rmsecv_min_new
                rms.append(rmsecv_min_new)

                stdout.write("\r"+str(i))
                stdout.write(" ")
                stdout.write(str(opt_comp_new))
                stdout.write(" ")
                stdout.write(str(rmsecv_min_new))
                stdout.flush()
            else:
                rms.append(rmsecv_min)

    stdout.write("\n")
    # print(np.sort(bands))
    # print('end')

    return bands, opt_comp,rms

#=======================================================================================================================
if __name__ == "__main__":

    data = pd.read_excel(r"/Users/macbook/folder/data/DD.xlsx").values
    y = data[:,0]
    X = data[:,1:]
    cars=CARS_Cloud(X, y)
    #print('改进算法选择出的波段{}\n主成分数{}'.format(cars[0][cars[1]],cars[2]))