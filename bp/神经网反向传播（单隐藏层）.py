# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 13:26:36 2017

@author: dell
"""




import numpy as np
from numpy.linalg import cholesky
import math

def dl_douto(data_label,outo,j):
    
     return (1.0 - data_label[j]) / outo[j] - data_label[j] / (1.0 - outo[j])
     
def douto_dneto(outo,j):
    
    return outo[j] * (1.0 - outo[j])

def dneto_douth(w_2,i):
    
    return w_2[i]
      
def  douth_dneth(outh,i,j):
    
    return outh[i][j] * (1.0 - outh[i][j])
    
def calculate_correct(data_label,outo):
    
    sum = 1.0
    
    for i in range(0,len(outo)):
        
        sum = sum * math.pow(outo[i] , 1 - data_label[i]) * math.pow(1 - outo[i] , data_label[i]) 
    
    return math.pow(sum,1.0/len(outo))
        
        
# 极大似然
def mlp(train_data,data_label,test_data,test_label):
    #从输入层到隐藏层
    w_1 = np.array([[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1]])
    #从隐藏层到输出层
    w_2 = np.array([0.1,0.1,0.1])
    #学习率
    step = 0.0001
    
    correct = 0.1
    
    while(True):
    
        #print w_1
        
        #print "###"

        neth = [[],[],[]]
    
        outh = [[],[],[]]
    
        neto = []
    
        outo = []  
    
        for i in range(0,len(train_data)):
        
            neth[1].append(np.dot(np.array(train_data[i]),w_1[0].T))
            neth[2].append(np.dot(np.array(train_data[i]),w_1[1].T))
            outh[0].append(1)        
            outh[1].append(1.0 / (1 + math.exp((-1)*neth[1][i])))
            outh[2].append(1.0 / (1 + math.exp((-1)*neth[2][i])))
        
            neto.append(outh[0][i] * w_2[0] + outh[1][i] * w_2[1] + outh[2][i] * w_2[2])
            outo.append(1.0 / (1 + math.exp((-1)*neto[i])))
        temp = calculate_correct(data_label,outo)
        
        #print temp
        
        if(temp - correct > 0.0000001 ):
                
            w_2new = list(w_2)
    
            for i in range(0,len(w_2new)):

                gradient = 0
        
                for j in range(0,len(train_data)):
            
                    gradient = gradient + dl_douto(data_label,outo,j) * douto_dneto(outo,j) * outh[i][j]
            
                #print gradient,i
        
                w_2new[i] = w_2new[i] + step * gradient
        
            for i in range(0,len(w_1)):
        
                for j in range(0,len(w_1[i])):
            
                    gradient = 0
            
                    for k in range(0,len(train_data)):
        
                        gradient = gradient + dl_douto(data_label,outo,k) * douto_dneto(outo,k) * dneto_douth(w_2,i+1) * douth_dneth(outh,i+1,k) * train_data[k][j]
                    
                    #print w_1[i][j],step*gradient
                    w_1[i][j] = (w_1[i][j] + step * gradient)  
                   # print w_1[i][j]
    
            w_2 = np.array(w_2new)
            
            correct = temp
        
        else:
            
            break
    
    count = 0
    
    
    for i in range(0,len(test_data)):
        
        neth1 = np.dot(np.array(test_data[i]),w_1[0].T)
        neth2 = np.dot(np.array(test_data[i]),w_1[1].T)
        outh0 = 1.0
        outh1 = 1.0 / (1 + math.exp((-1)*neth1))
        outh2 = 1.0 / (1 + math.exp((-1)*neth2))
        neto = outh0 * w_2[0] + outh1 * w_2[1] + outh2 * w_2[2]
        outo = 1.0 / (1 + math.exp((-1)*neto))
        
        if(outo >= 0.5 and test_label[i] == 0):
            
            count = count + 1
            
        elif(outo < 0.5 and test_label[i] == 1):
            
            count = count + 1
            
    print count
            
    print count/2000.0
    
def calculate_wtx(data,w,w0):
    sum = w0*1 + 0.0
    for i in range(0,len(data)):
        sum = sum + data[i]*w[i]
    return sum
    
def calculate_py1(data,w,w0):
    wtx = calculate_wtx(data,w,w0)
    return 1-1.0/(1+math.exp(wtx))            
def calcul_correct(w0,w,test_data,test_label):
    count = 0.0
    for i in range(0,len(test_data)):
        py_1 = calculate_py1(test_data[i],w,w0)
        if(py_1 >= 0.5 and test_label[i] == 1):
            count = count + 1
        elif(py_1 < 0.5 and test_label[i] == 0):
            count = count + 1
    print "测试样本正确率",count/len(test_data)            
            
        
        
def train_w_and_test_nopunish_graident(train_data,data_label,test_data,test_label,lamda):
    
    #自己生成的数据的w初始值
    w0 = 0
    w  = [0.5,0.5,0.5,0.5]
    #w=[0,0,0,0,0,0]
    #uci的数据的w初始值
    #w0 = 0.0
    #w=[0,0,0,0]
    l = -1e10
    while(True):
        #print "----"
        w0_temp = w0
        w_temp = [1,1,1,1]
        for i in range(0,len(w)):
            w_temp[i] = w[i]
        for i  in range(0,len(train_data)):
            py_1 = calculate_py1(train_data[i],w_temp,w0_temp)
            #对自己生成的数据，参数调整 
            w0 = w0 + 0.00001*(data_label[i] - py_1)
            #w0 = w0 + 0.0000005*(data_label[i] - py_1)
            #对uci的数据，参数调整
            #w0 = w0 + 0.000000003*(data_label[i] - py_1)
        w0 = w0 - lamda*w0_temp
        for i in range(0,len(w)):
            for j in range(0,len(train_data)):
                py_1 = calculate_py1(train_data[j],w_temp,w0_temp)
                #自己生成的数据，参数调整
                w[i] = w[i] + 0.00001*(train_data[j][i] * (data_label[j] - py_1)) 
                #对uci的数据，参数调整
               # w[i] = w[i] + 0.0000005*(train_data[j][i] * (data_label[j] - py_1))
                #w[i] = w[i] + 0.000000003*(train_data[j][i] * (data_label[j] - py_1)) 
        #print w0,w
        for i in range(0,len(w)):
            w[i] = w[i] - w_temp[i]*lamda
        sum = 0
        for i in range(0,len(train_data)):
            wtx = calculate_wtx(train_data[i],w,w0)
            sum = sum + data_label[i]*wtx - math.log(1+math.exp(wtx),math.e)
        correct_p = math.pow(math.exp(sum),1.0/len(train_data))
        #print "训练数据拟合率",correct_p
        #correct_p = math.pow(math.exp(sum),1.0/600)
        #print "训练数据拟合率",correct_p
        #真是数据为0.000001
        if(correct_p - l < 0.000001 ):
            calcul_correct(w0,w,test_data,test_label)
          #  print w0,w
            break;
        else:
            l = correct_p        
        
        
        
        
        

        
    
    







"""
train_data = []
data_label = []
test_data = []
test_label = []
f = open('test.txt','r')
dataset = f.readlines()
count = 0
for data in dataset:
    count = count + 1
    list_temp = []
    l = data.split(',')
    if(count < 200):
        for i in range(0,6):
            list_temp.append(float(l[i]))
        train_data.append(list_temp)
        data_label.append(int(l[6])-1)
    else:
        for i in range(0,6):
            list_temp.append(float(l[i]))
        test_data.append(list_temp)
        test_label.append(int(l[6])-1)
train_data = np.array(train_data)
data_label = np.array(data_label)
test_data  = np.array(test_data)
test_label = np.array(test_label)
list_1 = []
for data in train_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_1.append(list_temp)
train_data = np.array(list_1)
list_2 = []
for data in test_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_2.append(list_temp)    
test_data = np.array(list_2)
print train_data[1]
"""
sigma=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
#sigma = np.array([[1,0.25,0.25,0.25],[0.25,1,0.25,0.25],[0.25,0.25,1,0.25],[0.25,0.25,0.25,1]])
R = cholesky(sigma)

#设置高斯分布的mu
mu_1 = np.array([[1,1,1,1]]);
mu_2 = np.array([[2.5,2.5,2.5,2.5]])

#生成4维高斯分布数据，两个类别
s_1 = np.dot(np.random.randn(50,4),R)+mu_1
s_2 = np.dot(np.random.randn(50,4),R)+mu_2
l_1 = np.zeros((1,50))
l_2 = np.ones((1,50))

#将数组拼接 形成训练数据 
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
list_2 = []
for data in l_1:
    for ele in data.flat:
        list_2.append(ele)
for data in l_2:
    for ele in data.flat:
        list_2.append(ele)
train_data = np.array(list_1)
data_label = np.array(list_2)
"""
list_1 = []
for data in train_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_1.append(list_temp)
train_data = np.array(list_1)
"""
s_1 = np.dot(np.random.randn(1000,4),R)+mu_1
s_2 = np.dot(np.random.randn(1000,4),R)+mu_2
l_1 = np.zeros((1,1000))
l_2 = np.ones((1,1000))

#将数组拼接 形成测试数据
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
list_2 = []
for data in l_1:
    for ele in data.flat:
        list_2.append(ele)
for data in l_2:
    for ele in data.flat:
        list_2.append(ele)
test_data = np.array(list_1)
test_label = np.array(list_2)
lamda = 0
print "---,logistic"
train_w_and_test_nopunish_graident(train_data,data_label,test_data,test_label,lamda)
list_1 = []
for data in train_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_1.append(list_temp)
train_data = np.array(list_1)

list_2 = []
for data in test_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_2.append(list_temp)    
test_data = np.array(list_2)
#print train_data
#print test_data[1]
print "---shenjiangwang"
mlp(train_data,data_label,test_data,test_label)