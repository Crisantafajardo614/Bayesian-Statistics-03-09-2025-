#Crisanta Rae M. Fajardo
#March 09, 2025

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
uniform_distribution = sts.uniform.pdf(mu) + 1

uniform_distribution = uniform_distribution / uniform_distribution.sum()
beta_distribution = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_distribution = beta_distribution / beta_distribution.sum()
plt.plot(mu, beta_distribution, label="Beta Distribution")
plt.plot(mu, uniform_distribution, label="Uniform Distribution")
plt.xlabel(r"Value of $\mu$ in meters")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


