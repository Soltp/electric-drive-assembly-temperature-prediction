import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data_RGCN import preprocess_data,load_water_data,ols_filter  # 导入数据
from olsrgcn import tgcnCell
from sklearn.linear_model import LinearRegression as OLS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.006, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 200, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 20, 'hidden units of gru.')
flags.DEFINE_integer('seq_len',5 , '  time length of inputs.')
flags.DEFINE_integer('pre_len', 20, 'time length of prediction.')
flags.DEFINE_integer('batch_size',1920, 'batch size.')
flags.DEFINE_string('dataset', 'Te', 'Full or Te.')
flags.DEFINE_string('model_name', 'tgcn21', 'tgcn301 or gru')

model_name = FLAGS.model_name
data_name = FLAGS.dataset
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units


data, adj = load_water_data()


df=data
df.drop('torque', axis=1, inplace=True)
extra_feats = {
    'i_s1': lambda x: x['i_d'] ** 2 + x['i_q'] ** 2,
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),
    'motor_speed1': lambda x: x['motor_speed'] ** 2,}
df = df.assign(**extra_feats)
data01=df
Fill_features= ['pm', 'stator_tooth', 'stator_yoke','i_d','i_q', 'stator_winding']
target_cols = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']
All_cols = ['pm', 'stator_tooth','stator_yoke','stator_winding', 'coolant','ambient' ,'i_s1','motor_speed','motor_speed1']
Non_cols=['coolant','ambient','i_s1','motor_speed','motor_speed1']
PROFILE_ID_COL = 'profile_id'
x_cols1 =  [x for x in df.columns.tolist() if x not in Fill_features + [PROFILE_ID_COL]]
cols_to_smooth = ['ambient', 'coolant']
smoothing_window = 100
orig_x = df.loc[:, cols_to_smooth]
x_smoothed = [x.rolling(smoothing_window,center=True).mean() for p_id, x in
              df[cols_to_smooth + [PROFILE_ID_COL]].groupby(PROFILE_ID_COL, sort=False)]
df.loc[:, cols_to_smooth] = pd.concat(x_smoothed).fillna(orig_x)
p_df_list = [meas.drop(PROFILE_ID_COL, axis=1).reset_index(drop=True)
             for _, meas in df[x_cols1 + [PROFILE_ID_COL]].groupby([PROFILE_ID_COL], sort=False)]

spans=[160,620,3360,6500]

df = pd.concat([df,pd.concat([ols_filter(p,spans) for p in p_df_list], ignore_index=True)],axis=1).dropna().reset_index(drop=True)
x_cols =  [x for x in df.columns.tolist() if x not in Fill_features + [PROFILE_ID_COL]]
y_cols = target_cols
####################OLS与TGCN测与训练数据集构建#################
OLSpre_profiles= [13,31,18,30,29,8,20,15,14,16,36,5,32,21,7,3,9,27,11,24,74,75,41,68,50,53,49,81,60, 65, 72]
trainset1 = df.loc[~df.profile_id.isin(OLSpre_profiles), :].reset_index(drop=True)
pretest=df.loc[df.profile_id.isin(OLSpre_profiles), :].reset_index(drop=True)
test_profiles = [60,65,72]
train_profiles = [p for p in OLSpre_profiles if p not in test_profiles]
test_profiles1 = [65]

x_train = trainset1.loc[:, x_cols]
y_train = trainset1.loc[:, target_cols]
x_test = pretest.loc[:, x_cols]
y_test = pretest.loc[:, target_cols]

scaler = StandardScaler()
y_scaler = StandardScaler()
x_train1 = pd.DataFrame(scaler.fit_transform(x_train), columns=x_cols)
y_train1 = pd.DataFrame(y_scaler.fit_transform(y_train), columns=y_cols)
x_test1 = pd.DataFrame(scaler.transform(x_test), columns=x_cols)

ols = OLS(fit_intercept=False)
ols.fit(x_train1, y_train1)

predc = ols.predict(x_test1)
predc = pd.DataFrame(y_scaler.inverse_transform(predc), columns=y_test.columns)

max_value=150
profile_id1 = pretest[['profile_id']]
data_nontem = pretest.loc[:, Non_cols]
scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_nontem)
data_nontem1  = pd.DataFrame(scaler1.fit_transform(data_nontem), columns=Non_cols)

predc1=predc/max_value
data10= data_nontem1.assign(**predc1)
Orig_tem = pretest.loc[:, target_cols]
Orig_tem1 = Orig_tem / max_value
data20 = data_nontem1.assign(**Orig_tem1)
data1 = data10.assign(**profile_id1)
data2 = data20.assign(**profile_id1)

trainX, trainY, testX, testY = preprocess_data(data1,data2, All_cols,target_cols,test_profiles1,train_profiles,seq_len, pre_len)
num_nodes = len(All_cols)
num_nodes1 = len(target_cols)
totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def tgcn21(_X, _weights, _biases):

    cell_1 = tgcnCell(gru_units,adj,num_nodes=num_nodes,num_nodes1=num_nodes1)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes1,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes1,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes1])
    return output, m, states
        

inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes1])

weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}

if model_name == 'tgcn21':
    pred,ttts,ttto = tgcn21(inputs, weights, biases)
y_pred = pred
label = tf.reshape(labels, [-1,num_nodes1])

lambda_loss = 0.5
delta = tf.constant(1.75)
delta1=1.75
huber_loss = tf.reduce_sum(tf.multiply(tf.square(delta), tf.sqrt(1. + tf.square((y_pred-label) / delta)) - 1.))
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss=tf.reduce_mean(huber_loss+Lreg)

error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
path1 = '%s_%s_lr%r_loss%r_delta%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,lambda_loss,delta1,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
path2=open(path+'/result.txt','w')
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred,test_error11 = [],[],[],[],[],[],[],[]

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],feed_dict = {inputs:mini_batch, labels:mini_label}) #用feed_dict以字典的方式填充占位
        train_label = np.reshape(mini_label, [-1, num_nodes1])
        train_label01 = train_label * max_value
        train_output01 = train_output * max_value

        rmse01, mae01, acc01, r2_score01, var_score01 = evaluation(train_label01, train_output01)
        batch_loss.append(loss1)
        batch_rmse.append(rmse01)

    loss2, rmse2, test_output = sess.run([loss, error, y_pred],feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY, [-1, num_nodes1])
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    rmse, mae, acc, r2_score, var_score = evaluation(test_label1, test_output1)
    test_loss.append(loss2)
    test_rmse.append(rmse)
    test_mae.append(mae)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    error11=[]
    for j11, col in enumerate(target_cols):
        error21 = max(abs(test_output1[:, j11] - test_label1[:, j11]))
        error11.append(error21)
        error001=max(error11)
    test_error11.append(error001)


    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'train_loss:{:.4}'.format(loss1),
          'test_rmse:{:.4}'.format(rmse),
          'test_loss:{:.4}'.format(loss2),
          'test_mae:{:.4}'.format(mae),
          'test_r2:{:.4}'.format(r2_score),
          'test_var:{:.4}'.format(var_score),
          'test_acc:{:.4}'.format(acc),
          'test_error:{:.4}'.format(error001))

b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_error11.index(np.min(test_error11))

test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/min_error_result.csv',index = False,header = False)
var1 = pd.DataFrame(test_label1)
var1.to_csv(path + '/min_error_label1.csv', index=False, header=False)

index2 = test_rmse.index(np.min(test_rmse))
test_result2 = test_pred[index2]
var2 = pd.DataFrame(test_result2)
var2.to_csv(path+'/min_RMSE_result.csv',index = False,header = False)

index3 = test_acc.index(np.max(test_acc))
test_result3 = test_pred[index3]
var3 = pd.DataFrame(test_result3)
var3.to_csv(path+'/max_Acc_result.csv',index = False,header = False)

index4 = test_mae.index(np.min(test_mae))
test_result4 = test_pred[index4]
var4 = pd.DataFrame(test_result4)
var4.to_csv(path+'/min_mae_result.csv',index = False,header = False)

test_nondata=data2.loc[data2.profile_id.isin(test_profiles1), Non_cols]
test_nondata.to_csv(path+'/test_nondata.csv',index = False,header = False)
print('min_rmse:%r'%(test_rmse[index]),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index],
      'min_error:%r'%test_error11[index])

