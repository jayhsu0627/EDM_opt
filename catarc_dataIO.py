# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:14:36 2018

@author: 49570

读取和写入数据
"""
#判断是否整数
def isInt(x):
    try:
        x=int(x)
        return isinstance(x,int)
    except ValueError:
        return False

#读取.csv文件中的数据
def dataReader(filename):
    '''
    从.csv文件中读取数据
    一般默认的.csv文件的内容为: 
        文件参数
        ...
        (序号  时间      位移   载荷)
        1, 0.00000, -0.00458, 17.56
        ...
    若不符合此格式可能产生读取错误! 
    参数列表: 
        filename  读取的文件名
    返回列表: 
        time_array  时间序列
        displacement  位移序列, 与时间序列相对应
        load  载荷序列, 与时间序列相对应
    '''
    
    import csv
    #read .csv file
    f_input=open(filename, encoding='utf-8')
    csv_reader=csv.reader(f_input)
    
    #arrays of displacement and load inputs
    time_array=[]
    displacement=[]
    load=[]

    #information


    for row in csv_reader:
        if isInt(row[0]):
            time_array.append(float(row[1]))
            displacement.append(float(row[2]))
            load.append(float(row[3]))
    
    f_input.close()
    return [time_array, displacement, load]

def setOutput(filename, N, t, X, y, num_input):
    '''
    将N个样本数据写到文件中
    输出为N+1行: 
        第一行为N, num_input
        第2~N行每行包含一个样本数据
    参数列表: 略
    '''
    
    #output
    import csv
    with open(filename, 'w', newline="") as f_out:
        #文件头
        csv_writer=csv.writer(f_out)
        csv_writer.writerow([N, num_input])
        #数据部分
        for line in range(N):
            thisline=X[line]
            thisline.insert(0, t[line])
            thisline.append(y[line])
            csv_writer.writerow(thisline)
        
    return

def setInput(filename):
    '''
    从文件中读取N个样本数据
    返回N, t, X, y
    返回列表: 略
    '''
    
    import csv
    with open(filename, 'r') as f_in:
        csv_reader=csv.reader(f_in)
        
        ii=0
        N=0
        num_input=0
        t=[]
        X=[]
        y=[]
        for line in csv_reader:
            if ii==0:
                N=int(line[0])
                num_input=int(line[1])
                if N==0:
                    break
                ii+=1
            else:
                t.append(float(line[0]))
                thisX=[]
                for i in range(num_input):
                    thisX.append(float(line[1+i]))
                X.append(thisX)
                y.append(float(line[num_input+1]))
            
    return [N, t, X, y, num_input]

def resultOutput(filename, y):
    '''
    输出预测的y值
    '''
    
    import csv
    with open(filename, 'w', newline="") as f_out:
        csv_writer=csv.writer(f_out)
        for i in range(len(y)):
            csv_writer.writerow([y[i]])
        
    return








