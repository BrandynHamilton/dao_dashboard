#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression


# In[2]:


def calculate_annual_return(yearly_data):
    start_price = yearly_data['price'].iloc[0]
    end_price = yearly_data['price'].iloc[-1]
    start_date = yearly_data.index[0]
    end_date = yearly_data.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25

    if years < 1:
        # For partial year, calculate simple return
        annual_return = (end_price - start_price) / start_price
    else:
        # For full year or more, calculate annualized return
        annual_return = (end_price / start_price) ** (1 / years) - 1

    return annual_return



# In[3]:


def month_to_quarter(month):
    if month <= 3:
        return 'Q1'
    elif month <= 6:
        return 'Q2'
    elif month <= 9:
        return 'Q3'
    else:
        return 'Q4'


# In[4]:


def calculate_beta(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model.coef_[0]


# In[ ]:




