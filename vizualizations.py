#!/usr/bin/env python
# coding: utf-8

# In[78]:


import import_ipynb
import pandas as pd
import requests
import numpy as np
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')

import apis
import makerdao
import Lido


# In[79]:


import matplotlib.pyplot as plt
from makerdao import *

data = [assets, abs(liabilities), abs(equity)]

labels = ['Assets', 'Liabilities', 'Eqity']

plt.pie(data, labels=labels, autopct='%1.1f%%')

plt.show


# In[80]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

# Seaborn style setting
sns.set(style="white")

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(assetsdf.index, assetsdf['balance'], color='royalblue', linewidth=2)
plt.plot(liabilitiesdf.index, abs(liabilitiesdf['balance']), color='red', linewidth=2)

# Remove gridlines
plt.grid(False)

# Formatting the y-axis
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Adding title and labels
plt.title("Value of Assets Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Balance", fontsize=12)

# Removing the top and right spines for a clean look
sns.despine()

# Show the plot
plt.show()


# In[81]:


plt.plot(abs(liabilitiesdf))
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))


# In[82]:


plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

plt.plot(abs(equitydf))


# In[83]:


plt.plot(tbill_decimals['value'])


# In[84]:


plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

plt.plot(monthly_stats['net_income'].iloc[:-1])


# In[85]:


BTC = yf.Ticker("BTC-USD")

# get all stock infov
BTC_history = BTC.history(period='max', interval='1mo')

BTC_history.index = pd.to_datetime(BTC_history.index)

filtered_BTC_history = BTC_history[(BTC_history.index >= "2019-12") & (BTC_history.index < "2023-11")]

filtered_BTC_history.tail()

formatbtc = filtered_BTC_history

formatbtc.shape
monthly_stats['lending_income'].shape

formatbtc


# In[86]:


filtered_income = monthly_stats[(monthly_stats.index >= '2019-12-01') & (monthly_stats.index <= '2023-10-01')]



# In[87]:


plt.plot(formatbtc['Close'])


# In[88]:


import matplotlib.pyplot as plt

# Assuming 'monthly_stats' and 'tbill_decimals' are defined DataFrames

# Set the tick label format for the current axis
plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

# Plot the first three datasets against the primary y-axis
line1, = plt.plot(filtered_income['lending_income'], label='Lending Income')
line2, = plt.plot(filtered_income['net_income'], label='Net Income')

# Create and label the secondary y-axis for the tbill_decimals
ax2 = plt.gca().twinx()
line3, = ax2.plot(tbill_decimals['value'], 'r-', label='Risk-Free Rate')
ax2.set_ylabel('Risk-Free Rate', color='r')

# Make the x-ticks rotated for better readability
plt.xticks(rotation=45)

# Manually create the legend
plt.legend([line1, line2, line3, line4], ['Lending Income', 'Net Income', 'Risk-Free Rate'])

# Optionally, you can set the legend location explicitly
# plt.legend([line1, line2, line3], ['Lending Income', 'Net Income', 'Risk-Free Rate'], loc='best')

# Save the figure
plt.savefig('mkrstmt')

# Show the plot
plt.show()


# In[97]:


import matplotlib.pyplot as plt

# Assuming 'monthly_stats' and 'tbill_decimals' are defined DataFrames

# Set the tick label format for the current axis
plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

# Plot the first three datasets against the primary y-axis
line5, = plt.plot(filtered_income['lending_income'], label='Lending Income')
line6, = plt.plot(filtered_income['net_income'], label='Net Income')

# Create and label the secondary y-axis for the tbill_decimals
ax2 = plt.gca().twinx()
line7, = plt.plot(formatbtc['Close'], label='BTC Price (USD)', color='yellow')
ax2.set_ylabel('BTC Price', color='b')

# Make the x-ticks rotated for better readability
plt.xticks(rotation=45)

# Manually create the legend
plt.legend([line5, line6, line7], ['Lending Income', 'Net Income','BTC Price'])

# Optionally, you can set the legend location explicitly
# plt.legend([line1, line2, line3], ['Lending Income', 'Net Income', 'Risk-Free Rate'], loc='best')

# Save the figure
plt.savefig('mkrstmt')

# Show the plot
plt.show()


# In[91]:


import matplotlib.pyplot as plt

# Assuming 'monthly_stats' and 'tbill_decimals' are defined DataFrames

# Set the tick label format for the current axis
plt.gca().ticklabel_format(style='plain', axis='both', useOffset=False, useLocale=False)

# Plot the first three datasets against the primary y-axis
line1, = plt.plot(monthly_stats['lending_income'], label='Lending Income')
line2, = plt.plot(monthly_stats['net_income'], label='Net Income')


# Create and label the secondary y-axis for the tbill_decimals
ax2 = plt.gca().twinx()
line3, = ax2.plot(tbill_decimals['value'], 'r-', label='Risk-Free Rate')
ax2.set_ylabel('Risk-Free Rate', color='r')

# Make the x-ticks rotated for better readability
plt.xticks(rotation=45)

# Manually create the legend
plt.legend([line1, line2, line3], ['Lending Income', 'Net Income', 'Risk-Free Rate'])

# Optionally, you can set the legend location explicitly
# plt.legend([line1, line2, line3], ['Lending Income', 'Net Income', 'Risk-Free Rate'], loc='best')

# Save the figure
plt.savefig('mkrstmt')

# Show the plot
plt.show()


# In[93]:


tbill_timeseries['value'].plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




