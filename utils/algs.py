import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import os
os.environ['R_HOME'] = '/global/homes/z/zhangtao/soft/miniconda3/envs/luffy/lib/R'
#from rpy2.robjects.packages import importr
#import rpy2.robjects as ro
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#pcalg = importr('pcalg')
import skccm as ccm
from skccm.utilities import train_test_split
import lingam

#sys.path.append("/global/homes/z/zhangtao/climate_causal/notears/notears/")
#import linear
#import utils
#import nonlinear
from sklearn import preprocessing
import multiprocessing as mp
#from loguru import logger

def calc_lag_corr(x,y,max_lag=1):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        corr = y[0] + x[0]
        return corr, corr
    else:
        max_corr = 0
        max_corr_p = 0
        ntime = len(x)
        for i in range(max_lag):
            corr, corr_p = stats.pearsonr(x[:ntime-i], y[i:])
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_p = corr_p

        return max_corr, max_corr_p

def calc_corr(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        corr = y[0] + x[0]
        return corr, corr
    else:
        corr, corr_p = stats.pearsonr(x, y)

        return corr, corr_p

def calc_reg(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        reg = y[0] + x[0]
        return reg, reg
    else:
        X = sm.add_constant(x)
        #X = np.column_stack((x, x))
        reg_mod = sm.OLS(y,X).fit()
        reg = reg_mod.params[1]
        #reg = reg_mod.params[1]
        #reg_p = reg_mod.pvalues[0] + reg_mod.pvalues[0]
        reg_p = reg_mod.pvalues[1]
        return reg, reg_p

def calc_lingam(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        lingam = y[0] + x[0]
        return lingam,lingam
    else:
        data = np.transpose(np.array([x, y]))
        ro.globalenv['aa'] = data
        ro.r('res <- lingam(aa)')
        ro.r('resmat <- as(res, "amat")')
        resmat = np.array(ro.r['resmat'])
        llingam = resmat[0,1]
        return resmat[0,1], resmat[1,0]

def calc_Directlingam(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        llingam = y[0] + x[0]
        return llingam,llingam
    else:
        X = pd.DataFrame({'x1':x,'x2':y})
        #model = lingam.DirectLiNGAM()
        #model = lingam.DirectLiNGAM()
        #model = lingam.RCD()
        model = lingam.ICALiNGAM()
        model.fit(X)
        am = model.adjacency_matrix_
        p = model.get_error_independence_p_values(X)
        return am[1,0],am[0,1]
        #return am,p


def calc_pc(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        lingam = y[0] + x[0]
        return lingam,lingam
    else:
        data = np.transpose(np.array([x, y]))
        ro.globalenv['aa'] = data
        ro.r('suffStat <- list(C = cor(aa), n = nrow(aa))')
        ro.r('varname <- c("a","b")')
        ro.r('pc.fit = pc(suffStat, indepTest=gaussCItest, labels=varname, alpha=0.05)')
        ro.r('pcmat <- as(pc.fit@graph, "matrix")')
        pcmat = np.array(ro.r['pcmat'])
        #print(pcmat)
        return pcmat[0,1], pcmat[1,0]

# def calc_granger(x,y,nlag):
#     if np.ma.is_masked(y) or np.ma.is_masked(x):
#         granger = y[0] + x[0]
#         return granger, granger
#     else:
#         data = np.transpose(np.array([y, x]))
#         granger_model = grangercausalitytests(data, nlag, verbose=False)
#         granger = granger_model[1][0]['ssr_chi2test'][1]

#         data = np.transpose(np.array([x, y]))
#         granger_model = grangercausalitytests(data, nlag, verbose=False)
#         granger_r = granger_model[1][0]['ssr_chi2test'][1]

#         return granger, granger_r
    
def calc_granger(data, nlag):
    
    def granger_kernel(x1,x2,nlag):
        data = np.transpose(np.array([x1,x2]))
        granger_model = grangercausalitytests(data, nlag, verbose=False)
        res = granger_model[1][0]['ssr_chi2test'][1]
        return res
       
    N = data.shape[1]
    causal = np.ones([N,N])
    
    for i in range(N):
        for j in range(N):
            if i != j:
                causal[i,j] = granger_kernel(data[:,i],data[:,j],nlag)
    
    return causal


def calc_notears(x,y):
    if np.ma.is_masked(y) or np.ma.is_masked(x):
        notears = y[0] + x[0]
        return notears, notears
    else:
        data = np.transpose(np.array([x, y]))         
        aa = linear.notears_linear(data, lambda1=0.1, loss_type='l2', max_iter=200, w_threshold=0.05)
        notears = aa[0,1]
        return aa[0,1], aa[1,0]

def calc_CCM(data):
    def CCM_kernel(x,y):
        lag = 1
        embed = 2
        e1 = ccm.Embed(x)
        e2 = ccm.Embed(y)
        X1 = e1.embed_vectors_1d(lag,embed)
        X2 = e2.embed_vectors_1d(lag,embed)
        
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

        CCM = ccm.CCM() #initiate the class

        #library lengths to test
        len_tr = len(x1tr)
        lib_lens = np.arange(10, len_tr, len_tr/20, dtype='int')

        #test causation
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

        sc1,sc2 = CCM.score()
        
        sc1_mean = np.mean(sc1)
        sc2_mean = np.mean(sc2)
        
        return sc1_mean,sc2_mean

    N = data.shape[1]
    causal = np.zeros([N,N])
    
    for i in range(N):
        for j in range(i+1,N):
            sx1,sx2 = CCM_kernel(data[:,i],data[:,j])
            if sx1 > sx2:
                causal[j,i] = 1
            else:
                causal[i,j] = 1
                
    return causal
    

def calc_corr_wrapper(args):
    return calc_corr(*args)
    #return calc_lag_corr(*args)

def calc_reg_wrapper(args):
    return calc_reg(*args)

def calc_notears_wrapper(args):
    return calc_notears(*args)

def calc_lingam_wrapper(args):
    #return calc_lingam(*args)
    return calc_Directlingam(*args)

def calc_pc_wrapper(args):
    return calc_pc(*args)

def calc_granger_wrapper(args):
    return calc_granger(*args)
    #return causality_test(*args)


def causal_algs_2d_to_2d(data,nlat,nlon,np=20):
    results={}

    pool = mp.Pool(np)
    logger.info("Calc correlation")
    r = pool.map(calc_corr_wrapper,data)
    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
    p = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
    results['correlation'] = a
    results['correlation_p'] = p
    
#    logger.info("Calc regression")
#    r = pool.map(calc_reg_wrapper,data)
#    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
#    p = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
#    results['reg'] = a
#    results['reg_p'] = p
#
#    logger.info("Calc notears linear")
#    r = pool.map(calc_notears_wrapper,data)
#    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
#    b = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
#    results['notears_linear'] = a
#    results['notears_linear_r'] = b 

    logger.info("Calc lingam")
    r = pool.map(calc_lingam_wrapper,data)
    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
    b = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
    results['lingam'] = a
    results['lingam_r'] = b
    
    logger.info("Calc Granger")
    r = pool.map(calc_granger_wrapper,data)
    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
    b = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
    results['granger'] = a
    results['granger_r'] = b
    
#    logger.info("Calc pc")
#    r = pool.map(calc_pc_wrapper,data)
#    a = np.ma.masked_array([c[0] for c in r]).reshape(nlat,nlon)
#    b = np.ma.masked_array([c[1] for c in r]).reshape(nlat,nlon)
#    results['pc'] = a
#    results['pc_r'] = b
        
    return results
