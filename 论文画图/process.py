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

def CARS_Cloud(X, y, N=10, f=20, cv=5):
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
        b2 = np.arange(x.shape[1])
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        x_init = X[cal_index,:]
        print(x.shape)
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        ycal = y[cal_index]
        print('xcal',xcal.shape)
        '''
        回归系数排序
        '''
        pls = PLSRegression(n_components=f, scale=False)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        hh = np.argsort(-b, axis=0)
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV_cars = Cross_Validation(xcal, ycal, rindex + 1, cv)
        '''
        VIP值排序
        '''
        pls = PLSRegression(n_components=f, scale=False)
        pls.fit(x[cal_index,:], ycal)
        W = pls.x_weights_
        T = pls.x_scores_
        P = pls.x_loadings_
        V = np.dot(np.dot(P, np.diag(1 / np.diag(np.dot(P.T, P)))), P.T)
        V = np.dot(V, W)
        s = np.diag(np.dot(np.dot(T.T, T), np.dot(W.T, W)))
        s_vip = np.sqrt(len(s) * s / np.sum(s))
        vip_scores = np.dot(V ** 2, s_vip)
        ss = np.argsort(-vip_scores, axis=0)
        hou_index,RMSECV_houxiang = houxiang_pls(x[cal_index,:][:,ss],ycal,wave_num,x[cal_index,:][:,ss])

        if RMSECV_houxiang<RMSECV_cars:
            RMSECV.append(RMSECV_houxiang)
            ll = []
            for i in range(x[:,:][:,ss][:,hou_index][:,:wave_num].shape[1]):
                for j in range(x_init.shape[1]):
                    if (xcal[:, i] == x_init[:, j]).all():
                        ll.append(j)
            WaveData.append(ll)
            Coeff.append(vip_scores[ss][hou_index][:wave_num])
            x = x[:,:][:,ss][:,hou_index][:,:wave_num]
            print('hou',x.shape)

        else:
            RMSECV.append(RMSECV_cars)
            ll = []
            for i in range(xcal.shape[1]):
                for j in range(x_init.shape[1]):
                    if (xcal[:, i] == x_init[:, j]).all():
                        ll.append(j)
            Coeff.append(beta)
            WaveData.append(ll)
            x = np.squeeze(x[:, :wave_num][:,hh],axis=2)
            print('cars',x.shape)

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
    plt.savefig(r'/Users/macbook/Documents/data/CARS.png')
    plt.show()
    return OptWave,MinIndex
#=======================================================================================================================
if __name__ == "__main__":
    data = pd.read_excel(r"/Users/macbook/Documents/data/FL199.xlsx").values
    y = data[:, 0]
    X = data[:, 1:]
    cars = CARS_Cloud(X, y)
    print('结果：', cars)