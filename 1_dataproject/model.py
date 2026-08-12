# Imported packages
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
N = 50000
e = ['short', 'medium', 'long']
p_e = [0.4, 0.35, 0.25]
S = {'short': 1, 'medium': 3, 'long': 5}
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

# Assigning an education to all individuals using the defined probabilities
education = rng.choice(
    e,
    size=N,
    p=p_e
)

# Checking that the share of individuals with each level of education is close to the defined probabilities
print("Short:", np.mean(education == 'short'))
print("Medium:", np.mean(education == 'medium'))
print("Long:", np.mean(education == 'long'))

# Assigning education specific initial human capital
h_e0 = {'short': 1, 'medium': 1.2, 'long': 1.55}

# Evolution of human capital
h = np.zeros(N)
for x in ["short", "medium", "long"]:
    h[education == x] = h_e0[x]

# Labour market: Define employed and unemployed
employed = np.zeros(N, dtype=bool)
previous_income = np.zeros(N)

# Creates empty lists to store income and ages over time later
income_over_time = []
unemployment_over_time = []
ages = []

# Life-cycle simulation:
for age in range(18, 65):

    # Education status
    in_education = np.zeros(N, dtype=bool)

    for x in e:
        in_education[education == x] = age < 18 + S[x]

    # Labour market transitions
    unemployed = ~employed & ~in_education

    unemployment_over_time.append(
    unemployed.copy()
)

    # Generates a random number for each indiviudal and checks if it is less than the job finding probability
    job_finding = rng.random(N) < lambda1
    employed[unemployed & job_finding] = True

    # Generates a random number for each indiviudal and checks if it is less than the job finding probability
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

    # Incorporates the equations from the exercise
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

#Unemployment in Steady State
# Convert unemployment data to numpy array
unemployment_over_time = np.array(
    unemployment_over_time
)

# Calculate simulated unemployment rate by age
unemployment_rate = np.mean(
    unemployment_over_time,
    axis=1
)

# Theoretical steady-state unemployment rate
u_ss = sigma / (sigma + lambda1)
print(
    "Theoretical steady-state unemployment rate:",
    u_ss
)

# Check selected ages
for age in [25, 35, 45, 55, 60]:
    age_index = ages.index(age)
    print(
        f"Age {age}: simulated unemployment rate = "
        f"{unemployment_rate[age_index]:.4f}"
    )

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

# Gini coefficient

def gini(incomes):

    incomes = np.asarray(incomes)

    incomes = np.sort(incomes)

    n = len(incomes)

    index = np.arange(1, n + 1)

    return (
        (2 * np.sum(index * incomes))
        / (n * np.sum(incomes))
        - (n + 1) / n
    )

# Test: uniform distribution on [0,1]

uniform_test = np.linspace(0, 1, 100000)

print("Gini uniform:", gini(uniform_test))

# Gini for full simulated sample

full_sample = income_over_time.flatten()

gini_full = gini(full_sample)

print("Gini coefficient, full sample:", gini_full)

# Lorenz curve

sorted_income = np.sort(full_sample)

cumulative_income = np.cumsum(sorted_income)

lorenz_curve = cumulative_income / cumulative_income[-1]

population_share = np.arange(1, len(sorted_income) + 1) / len(sorted_income)

plt.figure(figsize=(8, 6))

plt.plot(
    population_share,
    lorenz_curve,
    label="Lorenz curve"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect equality"
)

plt.xlabel("Cumulative population share")
plt.ylabel("Cumulative income share")
plt.title("Lorenz Curve")

plt.legend()
plt.grid(True)

plt.show()

# Gini coefficient for each age

gini_by_age = []

for i, age in enumerate(ages):

    income_at_age = income_over_time[i]

    gini_age = gini(income_at_age)

    gini_by_age.append(gini_age)

    print(
        "Age:", age,
        "Gini:", gini_age
    )

# Plot Gini coefficient over the life cycle
plt.figure(figsize=(10, 6))

plt.plot(
    ages,
    gini_by_age,
    marker="o"
)

plt.xlabel("Age")
plt.ylabel("Gini coefficient")
plt.title("Income Inequality over the Life Cycle")

plt.grid(True)

plt.show()

# Alternative simulations

def simulate_model(
    education_differences=True,
    human_capital_shocks=True,
    unemployment_depreciation=True,
    unemployment=True
):

    # Education
    if education_differences:
        education = rng.choice(
            e,
            size=N,
            p=p_e
        )
    else:
        education = np.array(['medium'] * N)

    # Initial human capital
    h = np.zeros(N)

    for x in e:
        h[education == x] = h_e0[x]

    # Labour market status
    employed = np.zeros(N, dtype=bool)
    previous_income = np.zeros(N)

    if not unemployment:
        employed[:] = True

    income_over_time = []

    # Simulation
    for age in range(18, 65):

        # Education

        in_education = np.zeros(N, dtype=bool)

        if education_differences:
            for x in e:
                in_education[education == x] = (
                    age < 18 + S[x]
                )
        else:
            in_education[:] = age < 18 + S['medium']

        # Labour market

        if unemployment:

            unemployed = ~employed & ~in_education

            job_finding = rng.random(N) < lambda1

            employed[unemployed & job_finding] = True

            job_separation = rng.random(N) < sigma

            employed[employed & job_separation] = False

        else:

            employed[:] = ~in_education

        # Human capital shock

        if human_capital_shocks:

            psi = np.exp(
                rng.normal(
                    -0.5 * sigma_psi**2,
                    sigma_psi,
                    size=N
                )
            )

        else:

            psi = np.ones(N)

        # Human capital

        h_old = h.copy()

        for x in e:

            employed_group = (
                (education == x)
                & employed
                & ~in_education
            )

            if education_differences:
                growth = delta_e[x]
            else:
                growth = delta_e['medium']

            h[employed_group] = (
                h_old[employed_group]
                * (1 + growth)
                * psi[employed_group]
            )

            unemployed_group = (
                (education == x)
                & ~employed
                & ~in_education
            )

            if unemployment_depreciation:

                h[unemployed_group] = (
                    h_old[unemployed_group]
                    * (1 - delta)
                    * psi[unemployed_group]
                )

            else:

                h[unemployed_group] = (
                    h_old[unemployed_group]
                    * psi[unemployed_group]
                )

        # Income
        y = np.zeros(N)

        y[in_education] = y_SU

        y[employed & ~in_education] = (
            h[employed & ~in_education]
        )

        unemployed = ~employed & ~in_education
        if unemployment:

            y[unemployed] = np.maximum(
                rho * previous_income[unemployed],
                y_subscribt
            )

        else:

            y[unemployed] = 0

        # Previous income
        previous_income[
            employed & ~in_education
        ] = y[
            employed & ~in_education
        ]
        # Save income
        income_over_time.append(y.copy())
    return np.array(income_over_time)



# Baseline
baseline_income = simulate_model()

baseline_gini_pooled = gini(
    baseline_income.flatten()
)

baseline_gini_age45 = gini(
    baseline_income[45 - 18]
)



# No educational differences
no_education = simulate_model(
    education_differences=False
)

gini_no_education_pooled = gini(
    no_education.flatten()
)

gini_no_education_age45 = gini(
    no_education[45 - 18]
)



# No human capital shocks
no_shocks = simulate_model(
    human_capital_shocks=False
)

gini_no_shocks_pooled = gini(
    no_shocks.flatten()
)

gini_no_shocks_age45 = gini(
    no_shocks[45 - 18]
)

# No depreciation while unemployed

no_depreciation = simulate_model(
    unemployment_depreciation=False
)

gini_no_depreciation_pooled = gini(
    no_depreciation.flatten()
)

gini_no_depreciation_age45 = gini(
    no_depreciation[45 - 18]
)

# No unemployment

no_unemployment = simulate_model(
    unemployment=False
)

gini_no_unemployment_pooled = gini(
    no_unemployment.flatten()
)

gini_no_unemployment_age45 = gini(
    no_unemployment[45 - 18]
)

# Results

print()
print("Gini coefficients")
print("-" * 60)
print("                         Pooled       Age 45")
print("-" * 60)

print(
    f"Baseline:                "
    f"{baseline_gini_pooled:.4f}       "
    f"{baseline_gini_age45:.4f}"
)

print(
    f"No education differences:"
    f"{gini_no_education_pooled:.4f}       "
    f"{gini_no_education_age45:.4f}"
)

print(
    f"No human capital shocks: "
    f"{gini_no_shocks_pooled:.4f}       "
    f"{gini_no_shocks_age45:.4f}"
)

print(
    f"No unemployment depreciation:"
    f"{gini_no_depreciation_pooled:.4f}       "
    f"{gini_no_depreciation_age45:.4f}"
)

print(
    f"No unemployment:         "
    f"{gini_no_unemployment_pooled:.4f}       "
    f"{gini_no_unemployment_age45:.4f}"
)

# Contribution to inequality

print()
print("Reduction in Gini relative to baseline")
print("-" * 60)

print(
    "Education differences:",
    baseline_gini_pooled - gini_no_education_pooled
)

print(
    "Human capital shocks:",
    baseline_gini_pooled - gini_no_shocks_pooled
)

print(
    "Unemployment depreciation:",
    baseline_gini_pooled - gini_no_depreciation_pooled
)

print(
    "Unemployment:",
    baseline_gini_pooled - gini_no_unemployment_pooled
)

# Simulation with age-dependent severe illness

# Number of individuals
N = 50000

# Random number generator
seed = 42
rng = np.random.default_rng(seed)


# Model parameters

e = ['short', 'medium', 'long']

p_e = [0.4, 0.35, 0.25]

S = {
    'short': 1,
    'medium': 3,
    'long': 5
}

h_e0 = {
    'short': 1.0,
    'medium': 1.2,
    'long': 1.55
}

delta_e = {
    'short': 0.01,
    'medium': 0.02,
    'long': 0.03
}

delta = 0.06
sigma_psi = 0.10
lambda1 = 0.60
sigma = 0.05
y_SU = 0.45
rho = 0.60
y_subscribt = 0.35

# Assign education

education = rng.choice(
    e,
    size=N,
    p=p_e
)

# Initial human capital

h = np.zeros(N)

for x in e:
    h[education == x] = h_e0[x]

# Initial labour market status

employed = np.zeros(N, dtype=bool)

previous_income = np.zeros(N)

# Store income

income_over_time = []
ages = []

# Simulation

for age in range(18, 65):

    # Education status

    in_education = np.zeros(N, dtype=bool)

    for x in e:
        in_education[education == x] = (
            age < 18 + S[x]
        )

    # Labour market transitions

    unemployed = ~employed & ~in_education

    job_finding = rng.random(N) < lambda1

    employed[
        unemployed & job_finding
    ] = True

    job_separation = rng.random(N) < sigma

    employed[
        employed & job_separation
    ] = False

    # Severe illness

    # Probability increases with age

    illness_probability = (
        0.001 + 0.0005 * (age - 18)
    )

    illness = (
        rng.random(N) < illness_probability
    )


    # Sick individuals leave employment

    employed[illness] = False


    # Illness reduces human capital by 10%

    h[illness] *= 0.90

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

        # Employed

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


        # Unemployed

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

    # Student grant

    y[in_education] = y_SU

    # Employment income

    y[
        employed & ~in_education
    ] = h[
        employed & ~in_education
    ]

    # Unemployment income

    unemployed = ~employed & ~in_education

    y[unemployed] = np.maximum(
        rho * previous_income[unemployed],
        y_subscribt
    )

    # Save previous income

    previous_income[
        employed & ~in_education
    ] = y[
        employed & ~in_education
    ]

    # Save income

    income_over_time.append(
        y.copy()
    )

    ages.append(age)

# Convert to numpy array

income_over_time = np.array(
    income_over_time
)

# Mean income by age

mean_income = np.mean(
    income_over_time,
    axis=1
)

# Gini function

def gini(incomes):

    incomes = np.sort(incomes)

    n = len(incomes)

    index = np.arange(
        1,
        n + 1
    )

    return (
        2 * np.sum(index * incomes)
        / (n * np.sum(incomes))
        - (n + 1) / n
    )

# Gini coefficients

gini_pooled = gini(
    income_over_time.flatten()
)

gini_age45 = gini(
    income_over_time[45 - 18]
)

gini_age60 = gini(
    income_over_time[60 - 18]
)

# Results

print(
    "Mean income, age 45:",
    mean_income[45 - 18]
)

print(
    "Mean income, age 60:",
    mean_income[60 - 18]
)

print(
    "Gini, all ages pooled:",
    gini_pooled
)

print(
    "Gini, age 45:",
    gini_age45
)

print(
    "Gini, age 60:",
    gini_age60
)

# Plot mean income

plt.figure(figsize=(10, 6))

plt.plot(
    ages,
    mean_income
)

plt.xlabel("Age")
plt.ylabel("Mean income")
plt.title(
    "Mean Income with Age-Dependent Severe Illness"
)

plt.grid(True)

plt.show()

# Effect of severe illness on income and inequality

import numpy as np
import matplotlib.pyplot as plt

# Simulation function

def simulate_illness_model(include_illness=True):

    rng = np.random.default_rng(42)

    # Education
    education = rng.choice(
        e,
        size=N,
        p=p_e
    )

    # Initial human capital
    h = np.zeros(N)

    for x in e:
        h[education == x] = h_e0[x]

    # Labour market
    employed = np.zeros(N, dtype=bool)

    previous_income = np.zeros(N)

    income_over_time = []

    illness_over_time = []

    # Simulate life cycle

    for age in range(18, 65):

        # Education status
        in_education = np.zeros(N, dtype=bool)

        for x in e:
            in_education[education == x] = (
                age < 18 + S[x]
            )

        # Labour market transitions

        unemployed = ~employed & ~in_education

        job_finding = rng.random(N) < lambda1

        employed[
            unemployed & job_finding
        ] = True

        job_separation = rng.random(N) < sigma

        employed[
            employed & job_separation
        ] = False

        # Severe illness

        if include_illness:

            # Probability increases with age
            illness_probability = (
                0.001
                + 0.0005 * (age - 18)
            )

            illness = (
                rng.random(N)
                < illness_probability
            )

        else:

            illness = np.zeros(N, dtype=bool)


        # Sick individuals cannot work
        employed[illness] = False

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

            # Employed
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


            # Unemployed
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

        # Effect of severe illness on human capital

        if include_illness:

            h[illness] *= 0.90

        # Income

        y = np.zeros(N)

        # Education
        y[in_education] = y_SU

        # Employment
        y[
            employed & ~in_education
        ] = h[
            employed & ~in_education
        ]


        # Unemployment
        unemployed = ~employed & ~in_education

        y[unemployed] = np.maximum(
            rho * previous_income[unemployed],
            y_subscribt
        )


        # Previous income
        previous_income[
            employed & ~in_education
        ] = y[
            employed & ~in_education
        ]


        # Save
        income_over_time.append(y.copy())

        illness_over_time.append(
            illness.copy()
        )


    return (
        np.array(income_over_time),
        np.array(illness_over_time)
    )

# Gini coefficient

def gini(incomes):

    incomes = np.asarray(incomes)

    incomes = np.sort(incomes)

    n = len(incomes)

    index = np.arange(1, n + 1)

    return (
        2 * np.sum(index * incomes)
        / (n * np.sum(incomes))
        - (n + 1) / n
    )

# Baseline: no severe illness

income_baseline, illness_baseline = simulate_illness_model(
    include_illness=False
)

# Model with severe illness

income_illness, illness = simulate_illness_model(
    include_illness=True
)

# Mean income

mean_baseline = np.mean(
    income_baseline,
    axis=1
)

mean_illness = np.mean(
    income_illness,
    axis=1
)

# Gini coefficients - pooled

gini_baseline_pooled = gini(
    income_baseline.flatten()
)

gini_illness_pooled = gini(
    income_illness.flatten()
)

# Gini coefficients - selected ages

gini_baseline_45 = gini(
    income_baseline[45 - 18]
)

gini_illness_45 = gini(
    income_illness[45 - 18]
)

gini_baseline_60 = gini(
    income_baseline[60 - 18]
)

gini_illness_60 = gini(
    income_illness[60 - 18]
)

# Income at selected ages

print()
print("MEAN INCOME")
print("-" * 50)

print(
    "Age 45 baseline:",
    mean_baseline[45 - 18]
)

print(
    "Age 45 with illness:",
    mean_illness[45 - 18]
)

print(
    "Age 60 baseline:",
    mean_baseline[60 - 18]
)

print(
    "Age 60 with illness:",
    mean_illness[60 - 18]
)

# Gini results

print()
print("GINI COEFFICIENT")
print("-" * 50)

print(
    "Pooled baseline:",
    gini_baseline_pooled
)

print(
    "Pooled with illness:",
    gini_illness_pooled
)

print(
    "Age 45 baseline:",
    gini_baseline_45
)

print(
    "Age 45 with illness:",
    gini_illness_45
)

print(
    "Age 60 baseline:",
    gini_baseline_60
)

print(
    "Age 60 with illness:",
    gini_illness_60
)

# Changes caused by illness

print()
print("EFFECT OF SEVERE ILLNESS")
print("-" * 50)

print(
    "Change in pooled Gini:",
    gini_illness_pooled
    - gini_baseline_pooled
)

print(
    "Change in Gini at age 45:",
    gini_illness_45
    - gini_baseline_45
)

print(
    "Change in Gini at age 60:",
    gini_illness_60
    - gini_baseline_60
)

print(
    "Change in mean income at age 45:",
    mean_illness[45 - 18]
    - mean_baseline[45 - 18]
)

print(
    "Change in mean income at age 60:",
    mean_illness[60 - 18]
    - mean_baseline[60 - 18]
)

# Illness probability over age

illness_probability_by_age = []

for age in range(18, 65):

    probability = (
        0.001
        + 0.0005 * (age - 18)
    )

    illness_probability_by_age.append(
        probability
    )

# Plot mean income

plt.figure(figsize=(10, 6))

plt.plot(
    range(18, 65),
    mean_baseline,
    label="Baseline"
)

plt.plot(
    range(18, 65),
    mean_illness,
    label="With severe illness"
)

plt.xlabel("Age")
plt.ylabel("Mean income")
plt.title("Effect of Severe Illness on Mean Income")

plt.legend()
plt.grid(True)

plt.show()

# Plot Gini by age

gini_baseline_by_age = []

gini_illness_by_age = []

for i in range(len(range(18, 65))):

    gini_baseline_by_age.append(
        gini(income_baseline[i])
    )

    gini_illness_by_age.append(
        gini(income_illness[i])
    )


plt.figure(figsize=(10, 6))

plt.plot(
    range(18, 65),
    gini_baseline_by_age,
    label="Baseline"
)

plt.plot(
    range(18, 65),
    gini_illness_by_age,
    label="With severe illness"
)

plt.xlabel("Age")
plt.ylabel("Gini coefficient")
plt.title("Effect of Severe Illness on Income Inequality")

plt.legend()
plt.grid(True)

plt.show()

# Plot illness probability

plt.figure(figsize=(10, 6))

plt.plot(
    range(18, 65),
    np.array(illness_probability_by_age) * 100
)

plt.xlabel("Age")
plt.ylabel("Probability of severe illness (%)")
plt.title("Probability of Severe Illness by Age")

plt.grid(True)

plt.show()