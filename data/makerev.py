import streamlit as st
from data.makerdao import historical_mk, dsr, dai, stablecoins, monthly_stats, revenue
import pandas as pd


filtered_dsr = dsr[dsr.index > '2020-11-17']
filtered_dai = dai[dai.index > '2020-11-17']
filtered_stables = stablecoins[stablecoins.index > '2020-11-17']

filtered_dsr = filtered_dsr.rename(columns={'balance':'dsr_balance'})
filtered_dai = filtered_dai.rename(columns={'balance':'dai_balance'})
filtered_stables = filtered_stables.rename(columns={'balance':'cash_balance'})

historical_mk.index = pd.to_datetime(historical_mk.index)
filtered_dsr.index = pd.to_datetime(filtered_dsr.index)
filtered_dai.index = pd.to_datetime(filtered_dai.index)
filtered_stables.index = pd.to_datetime(filtered_stables.index)

# Merging historical_mk with filtered_dsr based on index
ev_df = historical_mk.merge(abs(filtered_dsr['dsr_balance']), left_index=True, right_index=True)

# Further merging the result with filtered_dai based on index
ev_df = ev_df.merge(abs(filtered_dai['dai_balance']), left_index=True, right_index=True)

ev_df = ev_df.merge(filtered_stables['cash_balance'], left_index=True, right_index=True)

historical_ev = ev_df['marketcap'] + (ev_df['dsr_balance'] + ev_df['dai_balance']) - ev_df['cash_balance']

st.line_chart(historical_ev)

monthly_ev = historical_ev.resample('M').mean()

monthly_ev.index = monthly_ev.index.normalize()

#print('resampled:',monthly_ev.resample('M').mean())
#print('not resampled:',monthly_ev)

date_fixed_income = monthly_stats['net_income'].resample('M').mean()

filtered_income = date_fixed_income[date_fixed_income.index >= '2020-12-31']

historical_ev = historical_ev.to_frame('historical_ev')

historical_ev_2 = historical_ev.merge(filtered_income.to_frame('net_income'), left_index=True, right_index=True)

filtered_revenue = revenue[revenue.index >= '2020-12-31']

ev_metrics = historical_ev_2.merge(filtered_revenue.to_frame('revenue'), left_index=True, right_index=True)

ev_metrics['ev_to_rev'] = ev_metrics['historical_ev'] / ev_metrics['revenue']

print(ev_metrics)

# Assuming historical_ev is your DataFrame and 'ev_to_rev' is the column with EV/Revenue ratios

threshold = 2000  # This is an example value; adjust it based on your analysis
ev_metrics['ev_to_rev_truncated'] = ev_metrics['ev_to_rev'].clip(upper=threshold)
ev_metrics['ev_multiple'] = ev_metrics['historical_ev'] / ev_metrics['net_income']

st.line_chart(ev_metrics['ev_to_rev_truncated'])
st.line_chart(ev_metrics['ev_multiple'])

print(ev_metrics)

