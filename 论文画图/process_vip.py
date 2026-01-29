import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy.io as scio
import pandas as pd
from sklearn.model_selection import cross_val_predict,LeaveOneOut,cross_val_predict
from sys import stdout
# 设置显示中文字体，
from pylab import mpl
import math
# 设置font.sans-serif 或 font.family 均可
mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  ## mac
plt.rcParams['font.family'] = ['Arial Unicode MS']  ## mac
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
import warnings

warnings.filterwarnings(action='ignore')  # 忽略告警
plt.rcParams['font.sans-serif']=['SimSun']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
#=======================================================================================================================
#========================================================后向选择=========================================================
def base_pls_cv(X, y, n_components, return_model=False):
    # Simple PLS
    pls_simple = PLSRegression(n_components=n_components)
    # Fit
    pls_simple.fit(X, y)
    # Cross-validation
    y_cv = cross_val_predict(pls_simple, X, y, cv=10)

    # Calculate scores
    score = r2_score(y, y_cv)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))

    if return_model == False:
        return (y_cv, score, rmsecv)
    else:
        return (y_cv, score, rmsecv, pls_simple)


def pls_optimise_components(X, y, npc):
    rmsecv = np.zeros(npc)
    for i in range(1, npc + 1, 1):
        # Simple PLS
        pls_simple = PLSRegression(n_components=i)
        # Fit
        pls_simple.fit(X, y)
        # Cross-validation
        y_cv = cross_val_predict(pls_simple, X, y, cv=10)

        # Calculate scores
        score = r2_score(y, y_cv)
        rmsecv[i - 1] = np.sqrt(mean_squared_error(y, y_cv))

    # Find the minimum of ther RMSE and its location
    opt_comp, rmsecv_min = np.argmin(rmsecv), rmsecv[np.argmin(rmsecv)]

    return (opt_comp + 1, rmsecv_min)

def houxiang_pls(X,y,n,init_x):
    Xr_optim = X.copy()
    wlr_optim = np.linspace(0, X.shape[1], X.shape[1])

    rmscv_min = 0.6 # Initialise to the baseline value
    #iter_max = n
    iter_max = n
    a = 15
    for rep in range(iter_max):
        rmscv = []
        r2 = []
        loo = LeaveOneOut()
        print(Xr_optim.shape)
        for train_wl, test_wl in loo.split(wlr_optim):

            opt_comp, rmsecv_min = pls_optimise_components(Xr_optim[:, train_wl], y, a if a<len(train_wl) else len(train_wl)-1)
            predicted, r2cv_loo, rmscv_loo = base_pls_cv(Xr_optim[:, train_wl], y, opt_comp)
            rmscv.append(rmscv_loo)
            r2.append(r2cv_loo)

            stdout.write("\r" + str(test_wl))
            stdout.write(" ")
            stdout.write(" %1.4f" % rmscv_loo)
            stdout.write(" ")
            stdout.write(" %1.4f" % r2cv_loo)
            stdout.write(" ")
            stdout.flush()

        new_rmscv = np.min(np.array(rmscv))
        stdout.write('\r')

        # print(rep, np.argmin(np.array(rmscv)), np.min(np.array(rmscv)), np.array(r2)[np.argmin(np.array(rmscv))] )
        print("Rep: %1d,  Deleted band: %1d, RMSCV: %1.4f, R^2: %1.4f " \
              % (rep, np.argmin(np.array(rmscv)), np.min(np.array(rmscv)), np.array(r2)[np.argmin(np.array(rmscv))]))
        if new_rmscv < rmscv_min:
            rmscv_min = new_rmscv
            Xr_optim = np.delete(Xr_optim, np.argmin(np.array(rmscv)), axis=1)
            wlr_optim = np.delete(wlr_optim, np.argmin(np.array(rmscv)))
        else:
            print("End of optimisation at step ", rep)
            break
    ll = []
    for i in range(Xr_optim.shape[1]):
        for j in range(init_x.shape[1]):
            if (Xr_optim[:,i]==init_x[:,j]).all():
                ll.append(j)
    print('errr',ll,len(ll))
    return ll,rmscv_min
#=======================================================================================================================
#========================================================Cars===========================================================
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

def CARS_Cloud(X, y, N=80, f=20, cv=5):
    p = 0.3
    m, n = X.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    x = copy.deepcopy(X)
    WaveData = []
    Coeff = []
    WaveNum =[]
    RMSECV = []
    r = []
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)
        x_init = X[cal_index,:]
        ycal = y[cal_index]
        if f>= x.shape[1]:
            f = x.shape[1]
        pls = PLSRegression(n_components=f, scale=False)
        pls.fit(x[cal_index,:], ycal)
        '''
        回归系数排序
        '''
        beta = pls.coef_
        b = np.abs(beta)
        hh = np.squeeze(np.argsort(-b, axis=0))
        x_cars = x[cal_index,:][:,hh][:,:wave_num]
        if f >= x_cars.shape[1]:
            f = x_cars.shape[1]
        rmsecv, rindex = PC_Cross_Validation(x_cars, ycal, f, cv)
        RMSECV_cars = Cross_Validation(x_cars, ycal, rindex + 1, cv)
        '''
        VIP值排序
        '''
        W = pls.x_weights_
        T = pls.x_scores_
        P = pls.x_loadings_
        V = np.dot(np.dot(P, np.diag(1 / np.diag(np.dot(P.T, P)))), P.T)
        V = np.dot(V, W)
        s = np.diag(np.dot(np.dot(T.T, T), np.dot(W.T, W)))
        s_vip = np.sqrt(len(s) * s / np.sum(s))
        vip_scores = np.dot(V ** 2, s_vip)
        ss = np.argsort(-vip_scores, axis=0)
        x_vip = x[cal_index, :][:, ss][:, :wave_num]
        if f >= x_vip.shape[1]:
            f = x_vip.shape[1]
        rmsecv, rindex = PC_Cross_Validation(x_vip, ycal, f, cv)
        RMSECV_vip = Cross_Validation(x_vip, ycal, rindex + 1, cv)

        if RMSECV_vip<RMSECV_cars:
            print('选择vip法')
            RMSECV.append(RMSECV_vip)
            ll = []
            for i in range(x_vip.shape[1]):
                for j in range(x_init.shape[1]):
                    if (x_vip[:, i] == x_init[:, j]).all():
                        ll.append(j)
            WaveData.append(ll)
            Coeff.append(vip_scores[ss][:wave_num])
            x = x[:, ss][:, :wave_num]

        else:
            print('选择回归系数法')
            RMSECV.append(RMSECV_cars)
            ll = []
            for i in range(x_cars.shape[1]):
                for j in range(x_init.shape[1]):
                    if (x_cars[:, i] == x_init[:, j]).all():
                        ll.append(j)
            WaveData.append(ll)
            Coeff.append(beta[hh][:wave_num])
            x = x[:,hh][:,:wave_num]

    COEFF = np.zeros((N,X.shape[1]))
    for i in range(N):
        for j in range(len(WaveData[i])):
            COEFF[i,WaveData[i][j]] = Coeff[i][j]

    MinIndex = np.argmin(RMSECV)
    OptWave = WaveData[MinIndex]

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(311)
    plt.xlabel('Monte Carlo iterations\n(a)', fontsize=fonts)
    plt.ylabel('Number of selected variables', fontsize=fonts)
    plt.title('The optimal number of iterations:' + str(MinIndex) + 'times', fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(312)
    plt.xlabel('Monte Carlo iterations\n(b)', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(313)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('各变量系数值', fontsize=fonts)
    plt.plot(COEFF)
    plt.vlines(MinIndex, -1e3, 1e3, colors='r')
    plt.show()
    plt.savefig(r'/Users/macbook/Downloads/CARS.png')
    return OptWave,MinIndex,np.min(RMSECV)
#=======================================================================================================================
if __name__ == "__main__":
    data = pd.read_excel(r"/Users/macbook/folder/data/pre/mscFLMA_N.xlsx").values
    y = data[:, 0]
    X = data[:, 1:]
    cars = CARS_Cloud(X, y)
    print('结果：', cars)