#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

beef = pd.read_csv('beef.csv')
beef


# In[4]:


import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# revised profit function
#profit = quantity * price - cost -> eq (3)


# In[15]:


# demand curve
sns.lmplot(x="Price", y="Quantity", data=beef, fit_reg=True) 


# In[13]:


# fit OLS model
model = ols("Quantity ~ Price", data = beef).fit()
# print model summary
print(model.summary())


# In[34]:


def calc_revenue(coef):
    price = 350 # example price  
    intercept = model.params['Intercept']
    quantity = intercept + coef*price
    revenue = quantity * price - 80 #fixed cost
    return revenue

# Then when calling it:

for coef in sims:
    rev_sims.append(calc_revenue(coef))
    rev_ci = np.percentile(rev_sims, [5, 95])  

import matplotlib.pyplot as plt

plt.hist(rev_sims, bins=20)
plt.axvline(rev_ci[0], c='r')
plt.axvline(rev_ci[1], c='r') 
plt.title("Simulated Revenue Distribution")
plt.show()


# In[24]:


# a range of diffferent prices to find the optimum one
Price = [320, 330, 340, 350, 360, 370, 380, 390]
# assuming a fixed cost
cost = 80
Revenue = []
for i in Price:
   quantity_demanded = 30.05 - 0.0465 * i
   
   # profit function
   Revenue.append((i-cost) * quantity_demanded)
# create data frame of price and revenue
profit = pd.DataFrame({"Price": Price, "Revenue": Revenue})
#plot revenue against price
plt.plot(profit["Price"], profit["Revenue"])


# In[22]:


# Find price and revenue for max revenue 
max_revenue = profit['Revenue'].max()
max_profit = profit[profit['Revenue'] == max_revenue]

print("{:<10} {:<10}".format('Price', 'Max Revenue')) 
print("{:<10} {:<10}".format(max_profit['Price'].values[0], max_profit['Revenue'].values[0]))


# In[37]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(beef, test_size=0.2, random_state=0)

model = ols("Quantity ~ Price", data=train).fit() 

test_preds = model.predict(test)

from sklearn.metrics import r2_score

r2 = r2_score(test['Quantity'], test_preds)

print("Out-of-sample R-Squared:")
print(r2)


# In[ ]:




