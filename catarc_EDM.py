# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:33:25 2018

@author: Zhang Yusi

catarc关于减震器或衬套的EDM(Empirical Dynamics Methods)模型
默认假设读取时域上的位移-载荷, 训练一个MLP(Multi-Layer Perceptron)进行拟合
自定义函数: 
    isInt(x)  判断x是否为整数, 用于.csv文件的数据读取
    csvReader(filename)  从.csv文件中读取时间、位移和载荷的数据
    setGeneration( , , )  用给定参数生成数据集
    nnTraining()  用给定参数设置神经网络, 然后训练
    outputFigure()  用给定参数输出结果的图像
"""

#从原始数据生成监督学习样本, 然后将样本划分为训练集, 验证集和测试集
def setGeneration(data, num_input=10, num_input_X=10, num_input_F=0, \
                  time_sample=[0., 60.], time_test=[20., 30.], ratio_valid=0.2):
    '''
    将位移的数据displacement按照输入参数生成样本, 并划分为训练集X_train, 验证集X_valid, 测试集X_test
    time_array以及load生成对应的t_*和y_*
    参数列表: 
        time_array  时间序列
        displacement  位移序列, 与时间序列相对应
        load  载荷序列, 与时间序列相对应
        num_input  默认为10, 样本数据的维度, 或用于神经网络的输入神经元个数
        num_input_After  默认为0, 表示样本选取当前时间点之后的数据点个数, 非0的值是非物理的! 
        time_sample  默认为[0,60], 函数将选取该时间段内的样本作为总样本
        time_test  默认为[20,30], 函数将选取该时间段内的样本作为训练集
        ratio_valid  默认为0.2, 验证集占总样本的比例
    返回列表：
        N_train  训练集样本数目
        t_train  时间序列
        X_train  输入, 每个点是i时刻及之前的num_input-1个时刻的位移构成的数组
        y_train  输出, 对应load
        N_valid  验证集样本数目, 由验证集占总样本的比例随机选取
        t_valid  验证集, 时间序列
        X_valid  验证集, 位移输入
        y_valid  验证集, load输出
        N_test   测试集样本数目, 截取一段时间内的数据作为测试
        t_test   测试集, 时间序列
        X_test   测试集, 位移输入
        y_test   测试集, load输出
    注记: 数据的scaler目前放在神经网络模块里进行
    '''
    [time_array, displacement, load]=data
    
    from random import sample as random_sample
    
    t=[]
    X=[]
    y=[]
    
    #inR=num_input_After
    in_X=num_input_X
    in_F=num_input_F
    
    for i in range(in_X, len(time_array)):
        if time_array[i]>=time_sample[0] and time_array[i]<time_sample[1]:
            t.append(time_array[i])
            X.append(displacement[i-in_X:i+1]+load[i-in_F:i])
            y.append(load[i])
        
    #divide arrays into sets
    N=len(t)
    
    t_train=[]
    X_train=[]
    y_train=[]
    
    t_valid=[]
    X_valid=[]
    y_valid=[]
    
    t_test=[]
    X_test=[]
    y_test=[]
    
    t_notest=[]
    X_notest=[]
    y_notest=[]
    
    #测试集
    for i in range(N):
        #choose data in 20.0s~30.0s
        if t[i]>=time_test[0] and t[i]<time_test[1]:
            t_test.append(t[i])
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            t_notest.append(t[i])
            X_notest.append(X[i])
            y_notest.append(y[i])
            
    N_test=len(t_test)
        
    #不用于测试的样本
    N_notest=len(t_notest)
    N_valid=int(ratio_valid*N)
    seq_valid=random_sample(range(N_notest), N_valid)
    seq_valid.sort()
    
    index=0
    for i in range(N_notest):
        if index<N_valid and i==seq_valid[index]:
            t_valid.append(t_notest[i])
            X_valid.append(X_notest[i])
            y_valid.append(y_notest[i])
            index+=1
        else:
            t_train.append(t_notest[i])
            X_train.append(X_notest[i])
            y_train.append(y_notest[i])
    
    N_train=len(t_train)
    
    #scaler
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    #fit training data
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    if N_valid!=0:
        X_valid=scaler.transform(X_valid)
    X_test=scaler.transform(X_test)
    
    #returns
    return [N_train, t_train, X_train, y_train, N_valid, t_valid, X_valid, y_valid, \
            N_test, t_test, X_test, y_test]

#用训练集、验证集训练MLP, 并给出在训练集上的结果
def nnTraining(N_train, t_train, X_train, y_train, nn_hidden_layer_sizes=[20, ], nn_max_iter=2000, optimizer=False):
    '''
    通过从filename中读入数据并生成训练集、验证集(暂无)和测试集, 进行神经网络的训练
    给出预测的值和均方根
    
    参数列表:  
        N_train 训练集样本数目
        t_train  时间序列
        X_train  输入, 每个点是i时刻及之前的num_input-1个时刻的位移构成的数组
        y_train  输出, 对应load
        nn_hidden_layer_sizes  MLP隐层神经元个数
        nn_max_iter  最大迭代次数
    返回列表: 
        mlp  训练后的神经网络
    ''' 
    
    from sklearn.neural_network import MLPRegressor
    hidden_layer_sizes=nn_hidden_layer_sizes[:]
    '''
    mlp=MLPRegressor(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=nn_max_iter, \
                     solver='adam', alpha=1e-6, \
                     learning_rate='adaptive', learning_rate_init=1e-3, tol=1e-6)
    '''
    
    from sklearn.model_selection import cross_val_score
    if optimizer:#隐层单元数的优化
        layerSize_opt=0
        score_opt=-1.
        for m in range(20, 40, 10):
            mlp=MLPRegressor(hidden_layer_sizes=(m, ), max_iter=nn_max_iter, \
                             solver='adam', alpha=1e-6, \
                             learning_rate='adaptive', learning_rate_init=1e-3, tol=1e-6)
            score=cross_val_score(mlp, X_train, y_train, cv=5)
            if layerSize_opt==0 or score.mean()<score_opt:
                score_opt=score.mean()
                layerSize_opt=m
                hidden_layer_sizes=[m, ]
                

    mlp=MLPRegressor(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=nn_max_iter, \
                     solver='adam', alpha=1e-6, \
                     learning_rate='adaptive', learning_rate_init=1e-3, tol=1e-6)
    mlp.fit(X_train, y_train)
    
    return [mlp, hidden_layer_sizes]

def modelSave(model, filename):
    from sklearn.externals import joblib
    f_name=filename+'_EDM.m'
    joblib.dump(model, f_name)
    
    return

def modelLoad(filename):
    from sklearn.externals import joblib
    f_name=filename+'_EDM.m'
    mlp=joblib.load(f_name)
    
    return mlp

#根据样本点X进行预测
def nnPredict(nn, X):
    '''
    根据训练好的神经网络进行预测
    参数列表: 
        nn  训练完成的MLPRegressor
        X   输入的样本点, 必须是一二维数组
    返回列表: 
        p_predict  预测值
    '''
    y_predict=nn.predict(X)
    return y_predict

#计算均方根误差
def rmsErrorCalculation(N, y, y_in):
    '''
    计算样本值和预测值之间的相对均方根误差
    参数列表: 
        N  样本数目
        y  样本输出值
        y_in  预测值
    返回列表: 
        error  相对均方根误差
    '''
    
    import math
    
    error=0.
    errorSquare=0.
    ySquare=0.
    
    for i in range(N):
        errorSquare+=(y_in[i]-y[i])**2/N
        ySquare+=y[i]**2/N
        
    error=math.sqrt(errorSquare)/math.sqrt(ySquare)
    
    return error

#输出图像
def figureOutput(fig_name, N, t, y, y_in, t_show, fig_linewidth=0.1, fig_dpi=600):
    '''
    计算样本值和预测值之间的相对均方根误差
    参数列表: 
        N  样本数目
        t  样本时间序列
        y  样本输出值
        y_in  预测值
    '''    
    
    import matplotlib.pyplot as plt
    
    t0=[]
    y0=[]
    y1=[]
    
    for i in range(N):
        if t[i]>=t_show[0] and t[i]<t_show[1]:
            t0.append(t[i])
            y0.append(y[i])
            y1.append(y_in[i])
        
    plt.plot(t0, y0, linewidth=fig_linewidth, color='black')
    plt.plot(t0, y1, linewidth=fig_linewidth, color='blue')
    plt.xlabel(u'Time(secs)', fontsize='small')
    plt.ylabel(u'N,N', fontsize='small')
    plt.grid(linestyle='--', linewidth=0.2, color='gray')
    plt.savefig(fig_name, dpi=fig_dpi)
    #plt.figure(facecolor='#0203e2')
    
    return

def spectrumOutput(fig_name, sampling_rate, N, t, t_show, y, y_in, fig_linewidth=0.1, fig_dpi=600):
    '''
    输出频谱
    参数列表: 
        同figureOutput
    '''
    
    import matplotlib.pyplot as spectrum
    import numpy as np
    
    fft_size=len(t)
    dt=t[fft_size-1]-t[0]
    sampling_rate=fft_size/dt
    
    y_f=np.fft.rfft(y)/fft_size
    y_in_f=np.fft.rfft(y_in)/fft_size
    
    freqs=np.linspace(0, dt*sampling_rate/2, fft_size/2+1)/dt
    
    y_fdb=20*np.log10(np.clip(np.abs(y_f), 1e-20, 1e200))
    y_in_fdb=20*np.log10(np.clip(np.abs(y_in_f), 1e-20, 1e200))
    
    #绘图
    spectrum.plot(freqs, y_fdb[0:len(freqs)], linewidth=fig_linewidth, color='black')
    spectrum.plot(freqs, y_in_fdb[0:len(freqs)], linewidth=fig_linewidth, color='blue')
    spectrum.ylim(-40, 40)
    spectrum.xlabel(u'Frequency(Hz)', fontsize='small')
    spectrum.ylabel(u'Amplitude(dB)', fontsize='small')
    spectrum.grid(linestyle='--', linewidth=0.2, color='gray')
    spectrum.savefig(fig_name, dpi=fig_dpi)
    
    return
    
    




