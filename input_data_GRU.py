import numpy as np
import pandas as pd

def load_water_data( ):
    adj = pd.read_csv(r'../data/TNT_adj.csv', header=None)
    adj = np.mat(adj)
    data = pd.read_csv(r'../data/measures.csv')
    return data, adj


def preprocess_data(data1,data2, All_cols,test_profiles1,train_profiles,seq_len, pre_len):
    trainX, trainY, testX, testY = [], [], [], []

    for i, (pid, x_train) in enumerate(data1.loc[data1.profile_id.isin(train_profiles), All_cols + ['profile_id']].groupby('profile_id')):
        end = x_train.shape[1]
        datax0 = np.mat(x_train, dtype=np.float32)
        datax1=np.delete(datax0, end-1, axis=1)
        for j in range(len(datax1) - seq_len - pre_len):
            a = datax1[j: j + seq_len + pre_len]
            trainX.append(a[0: seq_len])
    for i1, (pid, x_train1) in enumerate(data2.loc[data2.profile_id.isin(train_profiles), All_cols + ['profile_id']].groupby('profile_id')):
        end = x_train1.shape[1]
        datax10 = np.mat(x_train1, dtype=np.float32)
        datax11=np.delete(datax10, end-1, axis=1)
        for j1 in range(len(datax11) - seq_len - pre_len):
            a1 = datax11[j1: j1 + seq_len + pre_len]
            trainY.append(a1[seq_len: seq_len + pre_len])
    for i2, (pid1, y_train) in enumerate(data1.loc[data1.profile_id.isin(test_profiles1), All_cols + ['profile_id']].groupby('profile_id')):
        end1 = y_train.shape[1]
        datay0 = np.mat(y_train, dtype=np.float32)
        datay1 = np.delete(datay0, end1-1, axis=1)
        for j2 in range(len(datay1) - seq_len - pre_len):
            b2 = datay1[j2: j2 + seq_len + pre_len]
            testX.append(b2[0: seq_len])
    for i3, (pid1, y_train1) in enumerate(data2.loc[data2.profile_id.isin(test_profiles1),  All_cols+ ['profile_id']].groupby('profile_id')):
        end1 = y_train1.shape[1]
        datay10 = np.mat(y_train1, dtype=np.float32)
        datay11 = np.delete(datay10, end1-1, axis=1)
        for j3 in range(len(datay11) - seq_len - pre_len):
            b3 = datay11[j3: j3 + seq_len + pre_len]
            testY.append(b3[seq_len: seq_len + pre_len])
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1

def ols_filter(_df,spans):
    max_lookback = max(spans)
    dummy = pd.DataFrame(np.zeros((max_lookback, len(_df.columns))),
                         columns=_df.columns)
    temperature_cols = [c for c in ['ambient', 'coolant'] if c in _df]
    dummy.loc[:, temperature_cols] = _df.loc[0, temperature_cols].values
    _df = pd.concat([dummy, _df], axis=0, ignore_index=True)

    ew_mean = [_df.ewm(span=lb).mean()
                   .rename(columns=lambda c: c+'_ewma_'+str(lb))
               for lb in spans]
    ew_std = pd.concat(
        [_df.ewm(span=lb).std().fillna(0).astype(np.float32)
             .rename(columns=lambda c: c+'_ewms_'+str(lb))
         for lb in spans], axis=1)

    concat_l = [pd.concat(ew_mean, axis=1).astype(np.float32),
                ew_std,
                ]
    ret = pd.concat(concat_l, axis=1).iloc[max_lookback:, :]\
        .reset_index(drop=True)
    return ret
    
