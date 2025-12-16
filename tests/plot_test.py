import numpy as np

import matplotlib.pyplot as plt

# Define the function c(φ)
def c(phi, k, sigma):
    return sigma * k * np.log(1 + np.exp(-phi / sigma))

# Create φ values
phi = np.linspace(-5, 5, 1000)

# Define different parameter combinations
parameters = [
    (10, 0.05, "k=10, σ=0.05"),
    (1, 1.0, "k=1, σ=1.0"),
    (2, 0.5, "k=2, σ=0.5"),
    (2, 1.0, "k=2, σ=1.0"),
    (0.5, 2.0, "k=0.5, σ=2.0"),
]

# Plot
plt.figure(figsize=(10, 6))
for k, sigma, label in parameters:
    c_values = c(phi, k, sigma)
    plt.plot(phi, c_values, label=label, linewidth=2)

plt.xlabel("φ", fontsize=12)
plt.ylabel("c(φ)", fontsize=12)
plt.title("c(φ) = σk log(1 + exp(-φ/σ))", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()