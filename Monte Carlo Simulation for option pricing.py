#!/usr/bin/env python
# coding: utf-8

# # Options pricing using Geometric Brownian Motion and Monte-Carlo simulation

# ## Importing used libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Paramaters for simulating GBM

# In[51]:


#Number of steps of our simulation :
n=1000
#Number of simulations :
M=500
#Time
T=1
#Volatily 
s=0.22
#Drift coefficient:
µ=0.3
#Initial price 
S0=180


# ## Simulation and plot

# In[52]:


#Calculating each time step 
t=T/n
#Simulating our GBM using numpy
S_t=np.exp((µ-s**2/2)*t+s*np.random.normal(0,np.sqrt(t), size=(M,n)).T) #Transposing our array
S_t = np.vstack([np.ones(M),S_t]) #Stacking our Vector S_t with array of 1 so that we have a new axis full of 1.
#Had we did it with np.zeros we would have get only 0 for every coefficient.
S_t=S0 * S_t.cumprod(axis=0) #cumulative product of our matrix on each row,
#meaning that we are multiplying each coefficient at every step t by S0
#printing one branch to see the actual result


# In[53]:


#We now want to convert our steps as a time interval
Time = np.linspace(0,T,n+1) #Getting an array from 0 to 1 (not putting n+1 result in not "round" numbers)
#To plot our results, we have to create a new array with the same format as S_t
S_tcopy=np.full(shape=(M,n+1),fill_value=Time).T

plt.plot(S_tcopy,S_t)
plt.xlabel('Time')
plt.ylabel('Price of the considered option')
plt.show()


# # Applying this to Google's stock :

# ## Importing data using yfinance

# In[43]:


goog = yf.Ticker('GOOGL')
history=goog.history(start='2023-01-10',end='2024-02-27')


# ## Taking a look at our imported stock prices

# In[44]:


plt.plot(history['Close'])
plt.xlabel('Date')
plt.ylabel('Stock prices')
plt.show()


# ## Monte Carlo Simulation 

# In[54]:


#number of steps
n=len(history)
#initializing our array, we before used numpy but since yfinance uses panda, we'll do the same.
S_t = pd.DataFrame(0,index=history.index,columns=list(range(1,M+1))) #Using the same M as before (M=1000) we create a matrix 
#this matrix will contain our stock prices
S_t.iloc[0] = history['Close'].iloc[0] #S_0
for i in range(1,n):
    dS=µ*t+s*np.sqrt(t)*np.random.randn(M) 
    S_t.iloc[i]=S_t.iloc[i-1]+S_t.iloc[i-1]*dS
#after vizualisation we obtain similar graphics, well done !
plt.plot(S_t)
plt.xlabel('Time')
plt.ylabel('Price of the considered option')
plt.show()


# ## Comparison with reality 

# In[65]:


#Mean and volatility
St_mean = S_t.mean(axis=1)
St_thmean = history['Close'].iloc[0] * np.exp(µ*np.arange(n)/n*2) 
#I will turn St_thmean wich is the theoritecal value into a pandaframe so that it's easier to compare 
St_thmean = pd.DataFrame(St_thmean, index=St_mean.index)

print('Expected value using Monte Carlo Method : ', St_mean.mean())
print('Expected value using Theory : ', St_thmean.mean())

figure = plt.figure(figsize=(20,10))
axe = figure.add_subplot(111)
plt.plot(St_mean)
plt.plot(St_thmean)
plt.xlabel(('Date'))
plt.ylabel('Expected Value')

K,r = 80 , 0.1

#Applying this to the Asian Arithmetic Option :
H = np.maximum(St_mean-K,0)
prime = np.exp(-r*T) * np.mean(H)
print("The premium is :",prime)


# ### Another example : The double barrier call

# #### Parameters

# In[33]:


S0, K, r, sigma = 100, 95, 0.01, 1
down, up = 50, 150
N = 100000
t1, t2, T = 0.25, 0.75, 1 


# #### Simulating our trajectories using 4 different instants (easily doable for n instants)

# In[69]:


W = np.sqrt([t1, t2-t1, T-t2]) * np.random.standard_normal((N,3)).cumsum(axis = 1)
t = np.array([t1, t2, T])
S = S0 * np.exp((r-sigma**2/2) * t + sigma * W)
print(S[1])


# #### Payoff

# In[70]:


I = ((S[:,:-1] >= down) & (S[:,:-1] <= up)).sum(1) == 2
H = np.maximum(S[:,-1] - K, 0) * I


# In[71]:


prime = np.exp(-r*T) * np.mean(H)
print(f'Premium of the double barrier call : {prime:5.3f}')


# In[ ]:




