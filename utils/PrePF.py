import numpy as np
from statsmodels.tsa.tsatools import detrend
from global_land_mask import globe

def calc_ocean_mask(lat, lon):
    lon_fmt = np.zeros_like(lon)
    for i in range(len(lon)):
        if lon[i] <= 180:
            lon_fmt[i] = lon[i]
        else:
            lon_fmt[i] = lon[i] - 360
    lon_grid, lat_grid = np.meshgrid(lon_fmt,lat)
    ocean_mask = globe.is_ocean(lat_grid, lon_grid)

    return ocean_mask


def calc_anorm(data):
    data_anorm = np.zeros_like(data)
    
    for i in range(12):
        data_climo = np.mean(data[i::12,:,:], axis=0)
        data_anorm[i::12,:,:] = data[i::12,:,:] - data_climo
        
    return data_anorm

def calc_anorm_stddata(data, data_cut):
    data_anorm = np.zeros_like(data)
    
    for i in range(12):
        data_climo = np.mean(data_cut[i::12,:,:], axis=0)
        data_anorm[i::12,:,:] = data[i::12,:,:] - data_climo
        
    return data_anorm

def calc_detrend(data):
    nlat = data.shape[1]
    nlon = data.shape[2]

    data_detrend = np.zeros_like(data)
    for i in range(nlat):
        for j in range(nlon):
            data_detrend[:,i,j] = detrend(data[:,i,j],order=2)

    return data_detrend

def expend_array(data, ntime,nlat,nlon):
    data_e = np.zeros([ntime,nlat,nlon])
    for i in range(nlat):
        for j in range(nlon):
            data_e[:,i,j] = data.reshape(-1)
            
    return data_e

def generate_lead_data(a, b, lag_time):
    '''
    a is a 1d array; b is a 2d array;
    a is lag, b is lead
    '''
    nrows = a.shape[0]

    row_deleted = np.arange(nrows-1-lag_time,nrows-1)
    a_lag = a[lag_time:]
    b_lag = b[:nrows-lag_time]

    return a_lag, b_lag

'''
a is a 1d array; b is a 2d array;
a is lead, b is lag
'''
def generate_lag_data(a, b, lag_time):
    nrows = a.shape[0]

    row_deleted = np.arange(nrows-1-lag_time,nrows-1)
    a_lag = np.delete(a, row_deleted, 0)

    b_lag = b[lag_time:,:,:]

    print(a_lag.shape)

    return a_lag, b_lag
