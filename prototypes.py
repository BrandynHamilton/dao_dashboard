#linear regression and r value for net income and interest rates
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
#We do .iloc[:-1] because is API and month is not done; we use previous full month
x = tbill_decimals['value'].iloc[:-1].values.reshape(-1, 1)
y = filtered_stats['net_income'].iloc[:-1].values

model = LinearRegression()
model.fit(x, y)


r_squared = model.score(x, y)

print("R^2:", r_squared)



"""


# Create a scatter plot of the original data
plt.scatter(x, y, color='blue', label='Data points')

# Predict y values for the given x values
y_pred = model.predict(x)

# Plot the regression line
plt.plot(x, y_pred, color='red', label='Regression line')

# Add labels and title (optional)
plt.xlabel('tbill rate')
plt.ylabel('net income')
plt.title('Linear Regression Analysis')

# Show the legend
plt.legend()

# Display the plot
plt.show()


# In[90]:


y2 = filtered_stats['net_income'].iloc[:-1].values
x2 = tbill_decimals['value'].iloc[:-1].values

from scipy.stats import pearsonr

correlation_coefficient, _ = pearsonr(x2, y2)
print(correlation_coefficient)

eth = yf.Ticker('ETH-USD')

eth_history = eth.history(period='max', interval='1mo')

eth_history.index = pd.to_datetime(eth_history.index)

eth_history['daily'returns']

filtered_eth = eth_history[(eth_history.index >= "2019-12") & (eth_history.index < "2023-11")]

"""

