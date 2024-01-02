import matplotlib.pyplot as plt
import numpy as np
from makerdao import current_risk_free, average_yearly_risk_premium, beta   

# Constants (replace with your actual values)
risk_free_rate = current_risk_free  
risk_premium = average_yearly_risk_premium  # Example: 10%
mkr_beta = beta   # Replace with your calculated beta

# Beta values for SML
betas = np.linspace(0.5, 2, 100)  # Adjust the range as needed

# SML Equation
expected_returns = risk_free_rate + betas * average_yearly_risk_premium


# Plotting the SML
plt.figure(figsize=(10, 6))
plt.plot(betas, expected_returns, label="Security Market Line")
plt.scatter(mkr_beta, risk_free_rate + beta * (market_return - risk_free_rate), color='red', label='DAO Token')

plt.xlabel("Beta (Systematic Risk)")
plt.ylabel("Expected Return")
plt.title("Security Market Line (SML) with DAO Token")
plt.axhline(y=risk_free_rate, color='grey', linestyle='--', label="Risk-Free Rate")
plt.legend()
plt.grid(True)
plt.show()
