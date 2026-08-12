# Imported packages
import numpy as np
import matplotlib.pyplot as plt


# Model parameters
N = 50000
e = ['short', 'medium', 'long']
p_e = [0.4, 0.35, 0.25]
S_e = [1, 3, 5]
h_e0 = [1, 1.2, 1.55]
delta_e = {
    'short': 0.01,
    'medium': 0.02,
    'long': 0.03
}
delta = 0.06
sigma_psi = 0.1
lambda1 = 0.6
sigma = 0.05
y_SU = 0.45
rho = 0.6
y_subscribt = 0.35

# Random number generator (random seed)
seed = 42
rng = np.random.default_rng(seed)

# Draw the shock
psi = np.exp(rng.normal(-0.5*sigma_psi**2, sigma_psi, size=N))
print(psi.mean())

# Assigning an education to all individuals
education = rng.choice(
    e,
    size=N,
    p=p_e
)

print("Short:", np.mean(education == 'short'))
print("Medium:", np.mean(education == 'medium'))
print("Long:", np.mean(education == 'long'))

# Lenght of the levels of education
S = {'short': 1, 'medium': 3, 'long': 5}

# Assigning education specific initial human capital
h_e0 = {'short': 1, 'medium': 1.2, 'long': 1.55}

# Evolution of human capital
h = np.zeros(N)
for x in ["short", "medium", "long"]:
    h[education == x] = h_e0[x]

# Define unemployed, employed and under education
employed = np.zeros(N, dtype=bool)
previous_income = np.zeros(N)

employed = np.zeros(N, dtype=bool)

# Simulate from age 18 to 65

income_over_time = []
ages = []

for age in range(18, 65):

    # Education status
    in_education = np.zeros(N, dtype=bool)

    for x in e:
        in_education[education == x] = age < 18 + S[x]

    # Labour market transitions
    unemployed = ~employed & ~in_education

    job_finding = rng.random(N) < lambda1
    employed[unemployed & job_finding] = True

    job_separation = rng.random(N) < sigma
    employed[employed & job_separation] = False

    # Human capital shock
    psi = np.exp(
        rng.normal(
            -0.5 * sigma_psi**2,
            sigma_psi,
            size=N
        )
    )

    # Human capital
    h_old = h.copy()

    for x in e:

        employed_group = (
            (education == x)
            & employed
            & ~in_education
        )

        h[employed_group] = (
            h_old[employed_group]
            * (1 + delta_e[x])
            * psi[employed_group]
        )

        unemployed_group = (
            (education == x)
            & ~employed
            & ~in_education
        )

        h[unemployed_group] = (
            h_old[unemployed_group]
            * (1 - delta)
            * psi[unemployed_group]
        )

    # Income
    y = np.zeros(N)

    y[in_education] = y_SU

    y[employed & ~in_education] = (
        h[employed & ~in_education]
    )

    unemployed = ~employed & ~in_education

    y[unemployed] = np.maximum(
        rho * previous_income[unemployed],
        y_subscribt
    )

    # Save previous income
    previous_income[employed & ~in_education] = (
        y[employed & ~in_education]
    )

    # Save income for this age
    income_over_time.append(y.copy())
    ages.append(age)

u_ss = sigma / (sigma + lambda1)

print("Theoretical unemployment rate:", u_ss)

# Mean and percentiles

income_over_time = np.array(income_over_time)

mean_income = np.mean(income_over_time, axis=1)

p10 = np.percentile(income_over_time, 10, axis=1)
p25 = np.percentile(income_over_time, 25, axis=1)
p50 = np.percentile(income_over_time, 50, axis=1)
p75 = np.percentile(income_over_time, 75, axis=1)
p90 = np.percentile(income_over_time, 90, axis=1)


# Plot

plt.figure(figsize=(10, 6))

plt.plot(ages, mean_income, label="Mean")
plt.plot(ages, p10, label="P10")
plt.plot(ages, p25, label="P25")
plt.plot(ages, p50, label="P50")
plt.plot(ages, p75, label="P75")
plt.plot(ages, p90, label="P90")

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Income over the life cycle")

plt.legend()
plt.grid(True)

plt.show()


# Histograms of income distribution at selected ages

selected_ages = [25, 35, 45, 60]

plt.figure(figsize=(12, 8))

for i, age in enumerate(selected_ages):

    age_index = ages.index(age)

    income_at_age = income_over_time[age_index]

    plt.subplot(2, 2, i + 1)

    plt.hist(
        income_at_age,
        bins=50,
        edgecolor="black"
    )

    plt.xlabel("Income")
    plt.ylabel("Number of individuals")
    plt.title(f"Income distribution at age {age}")

    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()