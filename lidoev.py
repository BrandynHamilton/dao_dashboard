import streamlit as st
from Lido import *

market_cap = ldo_market_cap.iloc[::-1]

# Ensure cash index is tz-naive
cash.index = pd.to_datetime(cash.index).tz_localize(None)
liabilities.index = pd.to_datetime(liabilities.index).tz_localize(None)

# Resampling
monthly_cap = market_cap.resample('M').mean()
cash = cash.resample('M').mean()
liabilities = liabilities.resample('M').mean() 

# Merging - ensuring that both indices are tz-naive
lido_ev_metrics = monthly_cap.merge(cash.to_frame('cash'), left_index=True, right_index=True)

lido_ev_metrics = lido_ev_metrics.merge(liabilities.to_frame('debt'), left_index=True, right_index=True)

lido_ev_metrics['historical_ev'] = lido_ev_metrics['marketcap'] + lido_ev_metrics['debt'] - lido_ev_metrics['cash']

st.line_chart(lido_ev_metrics['historical_ev'])

print(lidoincome_df)

revenue = lidoincome_df['($) >Net Revenue'] - lidoincome_df['($) >Cost of Revenue']

revenue = revenue.resample('M').mean()

resampled_rev = revenue.iloc[::-1]


lido_ev_metrics = lido_ev_metrics.merge(resampled_rev.to_frame('revenue'),left_index=True, right_index=True)

print(lido_ev_metrics)

lido_ev_metrics['ev_to_rev'] = lido_ev_metrics['historical_ev'] / lido_ev_metrics['revenue']

filtered_ev_metrics = lido_ev_metrics[lido_ev_metrics.index > '2021-04-30']

st.line_chart (filtered_ev_metrics['ev_to_rev'])
print(filtered_ev_metrics)


