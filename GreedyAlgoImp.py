# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:39:13 2018

@author: barna
"""


import numpy as np
import sklearn.cluster as cl
import math as ma
import networkx as nx
import scipy.optimize as sc
import random
#This procedure will find the initial number of communities
def InitialCommunities(A,N,Tot):
 
 k=np.zeros(Tot,dtype=int)
 for t in range(0,Tot):
    
     adj_mat=A[t,:,:] 
     
     zero_matrix=np.zeros([N,N],dtype=int)
     identity_matrix=np.identity(N,dtype=int)
     degree=np.zeros(N,dtype=int)
    
     for i in range(0,N):
       for j in range(0,N):
          if(adj_mat[i][j]==1):
             degree[i]=degree[i]+1
     
     diag=np.diag(degree)
     sub_mat=diag-identity_matrix
     b_matrix=np.empty([2*N,2*N],dtype=int)
     b_matrix[0:N,0:N]=zero_matrix
     b_matrix[N:2*N,N:2*N]=adj_mat
     b_matrix[0:N,N:2*N]=sub_mat
     b_matrix[N:2*N,0:N]=np.negative(identity_matrix)
     eigenvalues=np.linalg.eigvals(b_matrix)
     frob_norm_sqrt=ma.sqrt(np.linalg.norm(b_matrix,ord=2))
    # print(frob_norm_sqrt)
     for e in range(0,eigenvalues.shape[0]):
         
         if(not(np.iscomplex(eigenvalues[e]))):
            
             if(abs(eigenvalues[e])>frob_norm_sqrt):
                 
                 k[t]=k[t]+1
 return k

#This generic method calculates the density of each community for a particular Adjacency Matrix and Community Assignment matrix
def GetDensity(numCom,commAssign,Adj_mat,N):
    Edges=np.zeros(numCom,dtype=int)
    #print(commAssign)
    for i in range(0,N):
        for j in range(i+1,N):
            if(Adj_mat[i,j]==1 and commAssign[i]==commAssign[j]):
              Edges[commAssign[i]]=Edges[commAssign[i]]+1
    
   
    Vertices=np.bincount(commAssign)
  #  print(Vertices)
    Density=np.zeros(numCom,dtype=float)
    for i in range(0,numCom):
        if(Edges[i]==0):
            Density[i]=0
        else:
            Density[i]=Edges[i]/Vertices[i] 
   
    return Density

#This method performs spectral clustering and some other functions
def PerformSpectralClustering(A,Tot,N,Density_Ratio,Comm_Density):
 K_init=InitialCommunities(A,N,Tot)
 NoOfComs=K_init
 
 CommAssign=np.zeros([Tot,N],dtype=int)

 # This for loops through all the graphs and computes the spectral clustering of each node and at last computes density of each community
 for t in range(0,Tot):
     spc=cl.SpectralClustering(n_clusters=K_init[t],affinity='precomputed', n_init=100)#
     spc.fit(A[t,:,:])
     CommAssign[t,:]=spc.labels_
     Comm_Density[t]=GetDensity(K_init[t],CommAssign[t,:],A[t,:,:],N) #=
   
    
              
 return [NoOfComs,CommAssign]


#This function gives the class for whose the value of Count of edges/Density is the maximum
def GetClassOnCondition(par1,par2,par3,c1,c2,c3):
    paralist=np.array([par1,par2,par3])
    classlist=np.array([c1,c2,c3])
    return (classlist[paralist.argsort()[paralist.size-1:paralist.size]])

#This function does the new assignments during the iterations.   
def GetNewAssignments(Adj_mat,n,cur_t,tot_t,ktot,CommAssign,Density_Ratio):
    for j in range(0,n):
        maxClass1=-1
        maxParam1=-1
        maxClass2=-1
        maxParam2=-1
        
        for k in range(0,ktot):
            cnt1=0
            cnt2=0
            cnt3=0
            parameter1=0
            parameter2=0
            parameter3=0
            for l in range(0,n):
            
                   if(cur_t>0):
                     
                     if(j!=l and Adj_mat[j,l]==1 and CommAssign[cur_t-1,l]==k):
                        cnt1=cnt1+1
                        #print("j={} l={} cur_t={} Comm_Density={} CommAssign={}".format(j,l,cur_t,Comm_Density[cur_t-1][k],CommAssign[cur_t-1,l]))
                        if(Comm_Density[cur_t-1][k]!=0):
                           parameter1=cnt1/Comm_Density[cur_t-1][k]
                        else:
                           parameter1=2*cnt1
                   if(cur_t<tot_t-1):
                     if(j!=l and Adj_mat[j,l]==1 and CommAssign[cur_t+1,l]==k):
                        cnt2=cnt2+1
                        if(Comm_Density[cur_t+1][k]!=0):
                           parameter2=cnt2/Comm_Density[cur_t+1][k]
                        else:
                           parameter2=2*cnt2
                   if(j!=l and Adj_mat[j,l]==1 and CommAssign[cur_t,l]==k):
                        cnt3=cnt3+1
                        if(Comm_Density[cur_t][k]!=0):
                           parameter3=cnt3/Comm_Density[cur_t][k]
                        else:
                           parameter3=2*cnt3
                       
                  
                   
                  
        
            if(parameter1>=Density_Ratio and parameter1>maxParam1):
                 maxParam1=parameter1
                 maxClass1=k
            if(parameter2>=Density_Ratio and parameter2>maxParam2):
                 maxParam2=parameter2
                 maxClass2=k
            
        if(maxClass1!=-1 and maxClass1==maxClass2):
           CommAssign[cur_t,j]=maxClass1
           Comm_Density[cur_t]=GetDensity(ktot,CommAssign[cur_t,:],Adj_mat,n)
        elif(maxClass1!=maxClass2 and CommAssign[cur_t,j]!=maxClass1 and CommAssign[cur_t,j]!=maxClass2):
         #  print('here1')
           Lr=GetClassOnCondition(parameter1,parameter2,parameter3,maxClass1,maxClass2,CommAssign[cur_t,j])
           if(Lr!=CommAssign[cur_t,j]):
              # print('here')
              # print(Lr)
               CommAssign[cur_t,j]=Lr
               
           Comm_Density[cur_t]=GetDensity(ktot,CommAssign[cur_t,:],Adj_mat,n)
           
        # This for loops through all the graphs and re computes the density based on the new class assignment           
    return CommAssign
        
        
           
        
        
                
def PerformIterations(n_com,comassign,Density_Ratio,Comm_Density):
 itr=1

 while(itr<=2):
    for i in range(0,timestamp):
        comassign=GetNewAssignments(A[i,:,:],nodes,i,timestamp,n_com[i],comassign,Density_Ratio)
    itr=itr+1
 return 


print("Start")
dense=3


timestamp=3
nodes=200
comms=4
#Adjacency Matrix for all time instances
A=np.empty([timestamp,nodes,nodes],dtype=int)

#Array representing probabilities of communities
commprobs=np.empty(comms,dtype=float)
#Initial communities
commprobs.fill(1/comms)



prop=np.empty([timestamp,nodes,comms],dtype=int)
prop[0,:,:]=np.random.multinomial(1,commprobs,size=nodes)
  
totalcnodes=0.2*nodes
c = list(range(0, nodes))

for i in range(1,timestamp):
    prop[i,:,:]=prop[i-1,:,:]
    change=random.sample(c,int(totalcnodes))
    ty=[prop[i-1,x,:] for x in change]
    nodescompleted=0
    
    while(nodescompleted<int(totalcnodes)):
        newcomm=np.random.multinomial(1,commprobs)
        if(np.array_equal(newcomm,ty[nodescompleted])):
            continue
        else:
            prop[i,change[nodescompleted],:]=newcomm
            nodescompleted=nodescompleted+1


pmat=np.zeros((comms,comms))
pmat.fill(0.02*dense)
#
np.fill_diagonal(pmat,np.multiply(dense,[0.1,0.1,0.1]))

probbig_arr=np.empty([timestamp,nodes,nodes])

for i in range(0,timestamp):
    for j in range(0,comms):
        for k in range(j,comms):
            lowv=pmat[j,k]-(i>0)*0.0049
            if(lowv<=0):
              lowv=0.1
            highv=pmat[j,k]+(i>0)*0.0050
            if(highv>=0.95):
              highv=0.95
              
            pmat[j,k]=np.random.uniform(low=lowv,high=highv,size=1)
            pmat[k,j]=pmat[j,k]
    #print(pmat)   
    probbig_arr[i]=np.dot(np.dot(prop[i,:,:],pmat),prop[i,:,:].transpose())

#print(probbig_arr)  
for i in range(0,timestamp):
    biggraph1=nx.Graph()
    biggraph1.add_nodes_from(range(nodes))
    for node1 in range(nodes):
      for node2 in range(node1+1,nodes):
        value = np.random.binomial(1,probbig_arr[i,node1,node2])
        if value > 0:
          biggraph1.add_edge(node1,node2)
#
    Adjmat=nx.to_numpy_matrix(biggraph1)
    A[i]=Adjmat


Comm_Density=[[]]*timestamp
Density_Ratio=0
[n_com,comassign]=PerformSpectralClustering(A,timestamp,nodes,Density_Ratio,Comm_Density)

t=np.copy(comassign)

PerformIterations(n_com,comassign,Density_Ratio,Comm_Density)
print(t)
print(comassign)
for i in range(0,t.shape[0]):
    print('here')
    cntf=0
    for j in range(0,t.shape[1]):
        if(t[i][j]!=comassign[i][j]):
            print(str('Spectral= '+str(t[i][j]))+" "+str('Greedy= '+str(comassign[i][j])))
            print(prop[i][j])
            cntf=cntf+1
    print(cntf)
