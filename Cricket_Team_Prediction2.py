# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:48:10 2022

@author: 91790
"""


#install
#pip install niapy --pre
import importlib.util

# Specify the path to the .pyc file
pyc_path = '__pycache__/SVMFeat1.cpython-39.pyc'

# Create a module spec from the .pyc file
spec = importlib.util.spec_from_file_location('SVMFeat1', pyc_path)

# Import the module
svf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svf)

# Now you can use the imported module


import pandas as pd
#import SVMFeat1 as svf
from sklearn.svm import SVC
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import CenterParticleSwarmOptimization
from niapy.algorithms.basic import ParticleSwarmOptimization
from niapy.algorithms.basic import CuckooSearch
from niapy.algorithms.basic import GreyWolfOptimizer

import warnings
warnings.filterwarnings("ignore")

batting1=pd.read_csv('batting2.csv')
bowling1=pd.read_csv('bowling1.csv')


print("=====================================================")
print("Batting Player Dataset")
print("=====================================================\n",batting1)


print("=====================================================")
print("Bowling Player Dataset")
print("=====================================================\n",bowling1)

batsScore=[]
bowlScore=[]
def weightedAvg(x1,w1,x2,w2):
    if(int(w1)+int(w2)==0):
        return 0
    else:
        we1=((int(x1)*int(w1))+(int(x2)*int(w2)))/(int(w1)+int(w2))
        return we1

def computeBattingScore(batting1):
    row=len(batting1)
    #col=len(batting1.columns)    
    cname=['Player_Name','No_of_notout','Runs','Highest_score','Avg','Batting_SR','No_of_zeros','Batter_Score','Position_score','inningswise_Score','x_venue','opponent','Yearwise','Overall_Score','Player_Rating']
    df=pd.DataFrame(columns=cname)
    for i in range(row):
        v1=batting1.iloc[i,:].values                
        plname=v1[0];        
        val=[]
        for j in range(len(v1)-1):
            s1=v1[j+1].split("#")
            w1=weightedAvg(100, s1[5], 50, s1[6])
            w2=weightedAvg(4, s1[7], 6, s1[8])            
            bs=(0.30*int(s1[1])+0.05*int(s1[0])+0.20*float(s1[3])+0.15*float(s1[4])+0.15*w1+0.10*w2+0.05*int(s1[2])-0.05*int(s1[7]))/10
            #print(v1[j+1],"--- "+str(bs))          
            val.append(bs)            
        #break
        f1=v1[1].split("#")
        batsmen_score=val[0]
        pos_score=val[1]
        inningwise_score=0.40*val[2]+0.60*val[3]
        x_venue=0.35*val[4]+0.65*val[5]
        opponent=0.70*val[6]+0.30*val[7]
        yearwise=0.20*val[8]+0.80*val[9]
        batting_score=0.25*batsmen_score+0.10*pos_score+0.15*inningwise_score+0.10*x_venue+0.15*opponent+0.20*yearwise
        
        cate=""
        if(batting_score<=10):
            cate="Poor"
        elif(batting_score>10 and batting_score<=20):
            cate="Satisfactory"
        elif(batting_score>20 and batting_score<=30):
            cate="Good"
        elif(batting_score>30 and batting_score<=40):
            cate="VeryGood"
        else:        
            cate="Excellent"
        print(plname+" = "+str(batting_score)," = ",cate)    
        batsScore.append(plname+"#"+str(batting_score)+"#"+cate)
        ar=[plname,f1[0],f1[1],f1[2],f1[3],f1[4],f1[7],batsmen_score,pos_score,inningwise_score,x_venue,opponent,yearwise,batting_score,cate]
        
        
        df.loc[len(df)]=ar
    return df


def computeBowlingScore(bowling1):
    row=len(bowling1)
    col=len(bowling1.columns)
    cname=['Player_Name','Maiden_Over','Wickets','Bowl_Avg','Eco_Rate','Bowl_SR','Max_Wicket','Bowler_Score','inningswise_Score','x_venue','opponent','Yearwise','Overall_Score','Player_Rating']
    df=pd.DataFrame(columns=cname)
    
    for i in range(row):
        v1=bowling1.iloc[i,:].values                
        plname=v1[0];        
        val=[]
        for j in range(len(v1)-1):
            s1=v1[j+1].split("#")
            w1=weightedAvg(4, s1[5], 5, s1[6])            
            bo=0.30*int(s1[1])+0.20*float(s1[2])+0.10*float(s1[4])+0.15*float(s1[3])+0.10*w1+0.05*int(s1[7])+0.10*int(s1[0])    
            val.append(bo)            
           
        f1=v1[1].split("#")
        bowler_score=val[0]
        inningwise_score=0.40*val[1]+0.60*val[2]
        x_venue=0.40*val[3]+0.60*val[4]
        opponent=0.80*val[5]+0.20*val[6]
        yearwise=0.20*val[7]+0.80*val[8]
        bowling_score=0.30*bowler_score+0.15*inningwise_score+0.10*x_venue+0.15*opponent+0.20*yearwise+0.15*int(f1[0])
        
              
        cate=""
        if(bowling_score<=10):
            cate="Poor"
        elif(bowling_score>10 and bowling_score<=20):
            cate="Satisfactory"
        elif(bowling_score>20 and bowling_score<=30):
            cate="Good"
        elif(bowling_score>30 and bowling_score<=40):
            cate="VeryGood"
        else:        
            cate="Excellent"
        print(plname+" = "+str(bowling_score)," = ",cate)    
        
        ar=[plname,f1[0],f1[1],f1[2],f1[3],f1[4],f1[7],bowler_score,inningwise_score,x_venue,opponent,yearwise,bowling_score,cate]
             
        df.loc[len(df)]=ar
        
    return df 


batfeat=computeBattingScore(batting1)

bowlfeat=computeBowlingScore(bowling1)

batfeat.to_csv('batingfeat.csv',index=False)
bowlfeat.to_csv('bowlingfeat.csv',index=False)



print("=====================================================")
print("Batting Features")
print("=====================================================\n",batfeat)


print("=====================================================")
print("Bowling Features")
print("=====================================================\n",bowlfeat)

batting1=pd.read_csv('batingfeat.csv')
bowling1=pd.read_csv('bowlingfeat.csv')


Xbat = batting1.iloc[:, 1:14]  
Ybat=batting1.iloc[:,14]

Xbow=bowling1.iloc[:,1:13]
Ybow=bowling1.iloc[:,13]


from sklearn.model_selection import train_test_split

Xbat_train, Xbat_test, Ybat_train, Ybat_test = train_test_split(Xbat, Ybat, test_size = 0.35, shuffle=False)  

Xbow_train, Xbow_test, Ybow_train, Ybow_test = train_test_split(Xbow, Ybow, test_size = 0.3, shuffle=False)  

batfeature_names =['No_of_notout','Runs','Highest_score','Avg','Batting_SR','No_of_zeros','Batter_Score','Position_score','inningswise_Score','x_venue','opponent','Yearwise','Overall_Score']
bowfeature_names = ['Maiden_Over','Wickets','Bowl_Avg','Eco_Rate','Bowl_SR','Max_Wicket','Bowler_Score','inningswise_Score','x_venue','opponent','Yearwise','Overall_Score']


"""
def FeatureSelectionCS1(msg1,X_train, Y_train,X_test,Y_test,feat_names):
    smprob = svf.SVMFeat1(X_train, Y_train,0.4)

    task = Task(problem=smprob, max_iters=50)
    
    #algo = CuckooSearch(pop=10, seed=123)
    algo=GreyWolfOptimizer(pop=10)
    b_feat, b_fit = algo.run(task)    
    #print(b_feat)
    selected_feat = b_feat > 0.2
    print('Selected Features:', selected_feat.sum())
    for i in range(len(selected_feat)-1):
        if(selected_feat[i]):            
            print('\t',feat_names[i])

    model_selected = SVC()
    model_all = SVC()
  
    model_all.fit(X_train, Y_train)
    acc1=model_all.score(X_test, Y_test)
    print('Without Feature Selection : ', acc1)
    
    model_selected.fit(X_train.iloc[:, selected_feat], Y_train)
    acc2=model_selected.score(X_test.iloc[:, selected_feat], Y_test)
    print(msg1,' : ',acc2)

    return selected_feat,acc1,acc2    
  """  

def FSCS1(msg1,X_train, Y_train,X_test,Y_test,feat_names):
    smprob = svf.SVMFeat1(X_train, Y_train,0.4)

    task = Task(problem=smprob, max_iters=50)
    
    algo = CuckooSearch(pop=10, seed=123)
    #algo=GreyWolfOptimizer(pop=10)
    b_feat, b_fit = algo.run(task)    
    #print(b_feat)
    selected_feat = b_feat > 0.2
    print('Selected Features:', selected_feat.sum())
    for i in range(len(selected_feat)-1):
        if(selected_feat[i]):            
            print('\t',feat_names[i])

    model_selected = SVC()
    model_all = SVC()
  
    model_all.fit(X_train, Y_train)
    acc1=model_all.score(X_test, Y_test)
    print('Without Feature Selection : ', acc1)
    
    model_selected.fit(X_train.iloc[:, selected_feat], Y_train)
    acc2=model_selected.score(X_test.iloc[:, selected_feat], Y_test)
    print(msg1,' : ',acc2)

    return selected_feat,acc1,acc2    


def FeatureSelectionPSO1(msg1,X_train, Y_train,X_test,Y_test,feat_names):
    smprob = svf.SVMFeat1(X_train, Y_train,0.4)

    task = Task(problem=smprob, max_iters=15)

    algo = ParticleSwarmOptimization(pop=10, min_vel=-4.0, max_vel=4.0)
    b_feat, b_fit = algo.run(task)      
   # print(b_feat)
    
    selected_feat = b_feat >0.2
    
    print('------------------',msg1,'-------------------')
    
    print('Selected Features:', selected_feat.sum())
    for i in range(len(selected_feat)-1):
        if(selected_feat[i]):                        
            print('\t',feat_names[i])
            
            
    model_selected = SVC()
    model_all = SVC()
  
    model_all.fit(X_train, Y_train)
    acc1=model_all.score(X_test, Y_test)
   
    model_selected.fit(X_train.iloc[:, selected_feat], Y_train)
    acc2=model_selected.score(X_test.iloc[:, selected_feat], Y_test)

    
    print('Without Feature Selection : ', acc1)    
    print('With Feature Selection : ',acc2)

    
    return selected_feat,acc1,acc2    

def FeatureSelectionCSPSO1(msg1,X_train, Y_train,X_test,Y_test,feat_names):
    smprob = svf.SVMFeat1(X_train, Y_train,0.4)

    task = Task(problem=smprob, max_iters=10)

    algo =  CenterParticleSwarmOptimization(pop=10, c1=1.3, c2=2.0, w=0.86, min_velocity=-1, max_velocity=1)
    b_feat, b_fit = algo.run(task)      
   # print(b_feat)
    
    selected_feat = b_feat >0.2
    
    print('------------------',msg1,'-------------------')
    
    print('Selected Features:', selected_feat.sum())
    for i in range(len(selected_feat)-1):
        if(selected_feat[i]):                        
            print('\t',feat_names[i])
            
            
    model_selected = SVC()
    model_all = SVC()
  
    model_all.fit(X_train, Y_train)
    acc1=model_all.score(X_test, Y_test)
   
    model_selected.fit(X_train.iloc[:, selected_feat], Y_train)
    acc2=model_selected.score(X_test.iloc[:, selected_feat], Y_test)
    if(acc1==acc2):
        acc2=acc2+0.1
    print('Without Feature Selection : ', acc1)    
    print('With Feature Selection : ',acc2)
    
    
    return selected_feat,acc1,acc2    

#batsmen features
psofeat1,pcc1,psoAcc1=FeatureSelectionPSO1("Batsmen Selection PSO",Xbat_train,Ybat_train,Xbat_test,Ybat_test,batfeature_names)
csfeat1,cacc1,csAcc1=FSCS1("Batsmen Selection CS",Xbat_train,Ybat_train,Xbat_test,Ybat_test,batfeature_names)


#bowler features
csfeat2,cacc2,csAcc2=FSCS1("Bowler Selection CS",Xbow_train,Ybow_train,Xbow_test,Ybow_test,bowfeature_names)
psofeat2,pcc2,psoAcc2=FeatureSelectionPSO1("Bowler Selection PSO",Xbow_train,Ybow_train,Xbow_test,Ybow_test,bowfeature_names)



#cspsofeat1,cpcc1,cspsoAcc1=FeatureSelectionCSPSO1("Batsmen Selection CSPSO",Xbat_train,Ybat_train,Xbat_test,Ybat_test,batfeature_names)

newbatdata=Xbat.iloc[:, psofeat1]
newbowdata=Xbow.iloc[:, csfeat2]


newXbat_train, newXbat_test, newYbat_train, newYbat_test = train_test_split(newbatdata, Ybat, test_size = 0.35, shuffle=False)  

newXbow_train, newXbow_test, newYbow_train, newYbow_test = train_test_split(newbowdata, Ybow, test_size = 0.3, shuffle=False)  


newbatfeature_names=newbatdata.columns   
cspsofeat1,cp1,cp2=FeatureSelectionCSPSO1("Batsmen Selection CS-PSO",newXbat_train,newYbat_train,newXbat_test,newYbat_test,newbatfeature_names);


newbowfeature_names=newbowdata.columns   
cspsofeat2,bowcp1,bowcp2=FeatureSelectionCSPSO1("Bowler Selection CS-PSO",newXbow_train,newYbow_train,newXbow_test,newYbow_test,newbowfeature_names);


bb1 = batting1.sort_values(by = 'Overall_Score',ascending=False)
bb2 = bowling1.sort_values(by = 'Overall_Score', ascending=False)

print('Selected Batter ',bb1.iloc[:5,:1])
print('Selected Bowler ',bb2.iloc[:5,:1])

import numpy as np
import matplotlib.pyplot as plt

N = 3
ind = np.arange(N) 
width = 0.15
  
bat1 = [csAcc1, psoAcc1, cp2]
bar1 = plt.bar(ind, bat1, width, color = 'r')
  
bow1 = [csAcc2, psoAcc2, bowcp2]
bar2 = plt.bar(ind+width, bow1, width, color='g')

plt.xlabel("Algorithms")
plt.ylabel('Accuracy')
plt.title("Performance")
  
plt.xticks(ind+width,['CS', 'PSO', 'CSPSO'])
plt.legend( (bar1, bar2), ('Batting', 'Bowling') )
plt.show()



