import pandas as pd
import numpy as np
import math as mt
def gradient_descent(x,y,):
    m_cur=b_cur=0
    learning_rate=0.005
    iteration=1000
    n=len(x)
    cost=0
    for i in range(iteration):
        y_pred=m_cur*x+b_cur
        prev_cost=cost
        cost=(1/n)*sum([val**2 for val in (y-y_pred)])
        md=-(2/n)*sum(x*(y-y_pred))
        bd=-(2/n)*sum(y-y_pred)
        m_cur=m_cur- learning_rate*md
        b_cur=b_cur- learning_rate*bd
        print(f"m: {m_cur}||b: {b_cur}||cost: {cost}||iteration: {i}")
        if(mt.isclose(cost,prev_cost,abs_tol=1e-6)): 
            print("Cost is close to previous cost, stopping iteration.")
            print(f"Final equation: cs = {m_cur:.4f} * math + {b_cur:.4f}")
            print(f"Final cost: {cost:.4f}")    
            return
df=pd.read_csv('test_scores.csv')
x=np.array(df['math'])
y=np.array(df['cs'])
gradient_descent(x,y)
#print(x)
#print(y)