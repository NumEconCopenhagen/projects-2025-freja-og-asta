# ---------------------------------- SIMULATING THE INCOME DISTRIBUTION ---------------------------------- #
# --------------------------------------------- 2.1 THE MODEL -------------------------------------------- #
# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Define model variables
N = 50000
e = ['short', 'medium', 'long']
p_e = [0.4, 0.35, 0.25]
S = {'short': 1, 'medium': 3, 'long': 5}
h_e0 = {'short': 1, 'medium': 1.2, 'long': 1.55}
delta_e = {'short': 0.01, 'medium': 0.02, 'long': 0.03}
delta = 0.0688
sigma_psi = 0.1
lambda1 = 0.6
sigma = 0.05
y_SU = 0.45
rho = 0.6
y_subscribt = 0.35

# Random number generator (random seed)
seed = 42
rng = np.random.default_rng(seed)

# Defining the human capital shock and testing if the mean is close to 1.
def test_shock():
    test_rng = np.random.default_rng(42)
    psi = test_rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, N)
    print("Mean of human capital shock:", psi.mean())

# Assigning an education to all individuals using the defined probabilities
education = rng.choice(e, size=N, p=p_e)

# Checking that the share of agents in the model with the different education lenghts is close to the given probabilities.
def education_shares(education):
    short_share = np.mean(education == 'short')
    medium_share = np.mean(education == 'medium')
    long_share = np.mean(education == 'long')
    return short_share, medium_share, long_share

# Start by creating a list of 50,000 zeros for each agent's initial human capital at age 18.
h = np.zeros(N)
# Then we assign each agent with the associated initial human capital they gain from their specfic education type.
for x in e:
    h[education == x] = h_e0[x]

# We define that every agent is unemployed when they are 18.
employed = np.zeros(N, dtype=bool)
# Next we define that everyones previous income is equal to zero at age 18.
previous_income = np.zeros(N)

#Assigns different income growth to the income depending on what education the agents have taken. 
delta_growth = np.array([delta_e[x] for x in education])

# Creates empty lists to store income and ages over time in the life-cycle simulation.
income_over_time = []
unemployment_over_time = []
ages = []

# --------------------------------------- LIFE CYCLE SIMULATION: --------------------------------------- #
for age in range(18, 65):

    # Assigning education status
    in_education = np.array([age < 18 + S[x] for x in education])

    # Defining unemployed as neither employed or in education
    unemployed = ~employed & ~in_education

    # Assigning employment status each period
    employed[unemployed & (rng.random(N) < lambda1)] = True

    # Assigning unemployment status each period
    employed[employed & (rng.random(N) < sigma)] = False

    # Redefining unemployed as neither employed or in education as changes was made for employed
    unemployed = ~employed & ~in_education
    unemployment_over_time.append(unemployed.copy())
    
    # Adding the human capital shock
    psi = np.exp(rng.normal(-0.5 * sigma_psi**2, sigma_psi, size=N))

    # Saving the value of human capital at the beginning of each age as it is needed in the equation for the evolution of human capital
    h_old = h.copy()

    employed_group = employed & ~in_education
    unemployed_group = ~employed & ~in_education

    # Incorporates the equations of the evolution of human capital from the exercise
    h[employed_group] = (h_old[employed_group] * (1 + delta_growth[employed_group]) * psi[employed_group])
    h[unemployed_group] = (h_old[unemployed_group] * (1 - delta) * psi[unemployed_group])

    # Income depending on employment and education
    y = np.zeros(N)
    y[in_education] = y_SU
    y[employed] = h[employed]
    y[unemployed] = np.maximum(rho * previous_income[unemployed],y_subscribt)

    # Save previous income
    previous_income[employed] = y[employed]

    # Save results
    income_over_time.append(y.copy())
    ages.append(age)

# ------------------------------------------------------------------------------------------------------ #
# -------------------------------- 2.2 SIMULATE THE INCOME DISTRIBUTION -------------------------------- #

# Convert unemployment data to a NumPy array
unemployment_over_time = np.array(unemployment_over_time)

# Calculate simulated unemployment rate for each age
unemployment_rate = np.mean(unemployment_over_time, axis=1)

# Calculate theoretical steady-state unemployment rate
def theoretical_unemployment():
    u_ss = sigma / (sigma + lambda1)
    return u_ss

# Get simulated unemployment rates at selected ages
def unemployment_at_ages(selected_ages):
    for age in selected_ages:
        age_index = ages.index(age)
        rate = float(unemployment_rate[age_index])
        print(f"Age {age}: {rate:.4f}")

# Check visually that the unemployment rate converges to the theoretical steady-state value
def plot_unemployment():
    plt.figure(figsize=(8, 4))
    plt.plot(ages, unemployment_rate, label="Simulated unemployment rate")
    plt.axhline(u_ss, color="red", linestyle="--", label="Theoretical steady-state")
    plt.xlabel("Age")
    plt.ylabel("Unemployment rate")
    plt.title("Unemployment Rate over the Life Cycle")
    plt.legend()
    plt.grid(True)

# Generating the mean and percentiles
income_over_time = np.array(income_over_time)
mean_income = np.mean(income_over_time, axis=1)
p10, p25, p50, p75, p90 = np.percentile(income_over_time, [10, 25, 50, 75, 90], axis=1)

# Plot the mean and percentiles
def plot_income_distribution():
    plt.figure(figsize=(8, 4))
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

# Plot the histograms at specific ages
def plot_income_histograms(selected_ages=[25, 35, 45, 60]):
    plt.figure(figsize=(10, 6))
    for i, age in enumerate(selected_ages):
        income = income_over_time[ages.index(age)]
        plt.subplot(2, 2, i + 1)
        plt.hist(income, bins=50, edgecolor="black")
        plt.xlim(0, 11)
        plt.xlabel("Income")
        plt.ylabel("Number of individuals")
        plt.title(f"Income distribution at age {age}")
        plt.grid(alpha=0.3)
    plt.tight_layout()

# ------------------------------------------------------------------------------------------------------ #
# ---------------------------------- 2.3 COMPUTE THE GINI COEFFICIENT ---------------------------------- #
# Calculating the Gini coefficient by sorting the income, giving each agent an index, and lastly calculating the Gini coefficient
def gini(incomes):
    incomes = np.sort(np.asarray(incomes))
    n = len(incomes)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * incomes)) / (n * np.sum(incomes)) - (n + 1) / n

# Testing the Gini calculation
def gini_uniform_test():
    test = np.linspace(0, 1, 100000)
    print("Gini uniform:", gini(test))

# Gini coefficient for the full sample
def gini_full_sample():
    return gini(income_over_time.flatten())

#Printing the Gini coefficient
def print_gini():
    print("Gini coefficient, full sample:", gini_full_sample())

# Gini coefficient by different age groups
def gini_by_age():
    gini_age = []
    for i, age in enumerate(ages):
        income_at_age = income_over_time[i]
        gini_age_value = float(gini(income_at_age))
        gini_age.append(gini_age_value)
        print(f"Age: {age:2d}   Gini: {gini_age_value:.4f}")

# Calculate Lorenz curve
def lorenz_curve():
    income = np.sort(income_over_time.flatten())
    cumulative_income = np.cumsum(income)
    lorenz = cumulative_income / cumulative_income[-1]
    population = np.arange(1, len(income) + 1) / len(income)
    return population, lorenz

# Plot Lorenz curve
def plot_lorenz_curve():
    population, lorenz = lorenz_curve()
    plt.figure(figsize=(6, 5))
    plt.plot(population, lorenz, label="Lorenz curve")
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect equality")
    plt.xlabel("Cumulative population share")
    plt.ylabel("Cumulative income share")
    plt.title("Lorenz Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------------------------------- #
# ------------------------------------- 2.4 WHAT DRIVES INEQUALITY? ------------------------------------- #

# Create four additional simulation
def simulate_model(
    education_differences=True,
    human_capital_shocks=True,
    unemployment_depreciation=True,
    unemployment=True):

    # Changes the education type for all agents in the model to medium.
    if education_differences:
        education = rng.choice(e, size=N, p=p_e)
        delta_growth = np.array([delta_e[x] for x in education])
    else:
        education = np.full(N, "medium")
        delta_growth = np.full(N, delta_e["medium"])

    # Assigning initial human capital
    h = np.array([h_e0[x] for x in education])

    # We define that every agent is unemployed when they are 18.
    employed = np.zeros(N, dtype=bool)

    # Next we define that everyones previous income is equal to zero at age 18.
    previous_income = np.zeros(N)

    # Creating a list to use when eleminating unemployment from the model.
    if not unemployment:
        employed[:] = True

    # Creates empty lists to store income the simulation.
    income_over_time = []

    # --------------------------------------- ALTERNATIVE SIMULATIONS --------------------------------------- #
    for age in range(18, 65):

        # Assigning education status
        in_education = np.array([age < 18 + S[x] for x in education])

        # Defining unemployed as neither employed or in education
        if unemployment:
            unemployed = ~employed & ~in_education

            # Assigning employment status each period
            employed[unemployed & (rng.random(N) < lambda1)] = True

            # Assigning unemployment status each period
            employed[employed & (rng.random(N) < sigma)] = False
            # Eliminating unemployment from the model
        else:
            employed[:] = ~in_education

        # Define employed and unemployed after labour market transitions
        employed_group = employed & ~in_education
        unemployed_group = ~employed & ~in_education

        # Adding the human capital shock
        if human_capital_shocks:
            psi = np.exp(rng.normal(-0.5 * sigma_psi**2, sigma_psi, N))
        else:
            psi = np.ones(N)

        # Save human capital before updating
        h_old = h.copy()

        # Adding depreciation equal to zero when simulating without depreciation
        depreciation = delta if unemployment_depreciation else 0

        # Incorporates the equations of the evolution of human capital from the exercise
        h[employed_group] = (h_old[employed_group] * (1 + delta_growth[employed_group]) * psi[employed_group])
        h[unemployed_group] = (h_old[unemployed_group] * (1 - depreciation) * psi[unemployed_group])

        # Income depending on employment and education
        y = np.zeros(N)
        y[in_education] = y_SU
        y[employed] = h[employed]

        # If unemployment is a part of the model, the income is as fraction of the previous income.
        if unemployment: y[unemployed_group] = np.maximum(rho * previous_income[unemployed_group], y_subscribt)

        # Save previous income
        previous_income[employed] = y[employed]

        # Save income
        income_over_time.append(y.copy())

    return np.array(income_over_time)

# ------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def run_alternative_simulations():
    # Defining the different models
    models = {
        "Baseline": {},
        "No education differences": {"education_differences": False},
        "No human capital shocks": {"human_capital_shocks": False},
        "No unemployment depreciation": {"unemployment_depreciation": False},
        "No unemployment": {"unemployment": False}}
    results = {}

    #Running the different models
    for name, options in models.items():
        income = simulate_model(**options)
        results[name] = {
            "Pooled": gini(income.flatten()),
            "Age 50": gini(income[ages.index(50)])
        }

    #Calculating the Gini coefficients
    print("Gini coefficients")
    for name, values in results.items():
        print(
            f"{name}: "
            f"Pooled = {values['Pooled']:.4f}, "
            f"Age 50 = {values['Age 50']:.4f}")

    #Comparing the Gini coefficients from the other models with the one from the baseline model
    baseline = results["Baseline"]["Pooled"]
    print("\nDifference in pooled Gini relative to baseline")

    for name, values in results.items():
        if name != "Baseline":
            difference = baseline - values["Pooled"]
            print(f"{name}: {difference:.4f}")

    return results

# ------------------------------------------------------------------------------------------------------ #
# -------------------------------------- 2.5 EXTENSION: MORE RISK -------------------------------------- #

# Simulation with age-dependent probability of illness
def simulate_illness_model(include_illness=True):

    # Assigning an education to all individuals using the defined probabilities
    education = rng.choice(e, size=N, p=p_e)

    #Assigns different income growth to the income depending on what education the agents have taken.
    delta_growth = np.array([delta_e[x] for x in education])

    # Initial human capital
    h = np.array([h_e0[x] for x in education])

    # We define that every agent is unemployed when they are 18.
    employed = np.zeros(N, dtype=bool)

    # Next we define that everyones previous income is equal to zero at age 18.
    previous_income = np.zeros(N)

    # Creates empty lists to store income in the simulation.
    income_over_time = []

  # -------------------------------------- HEALTH SHOCK SIMULATIONS ------------------------------------- #
    for age in range(18, 65):

        # Assigning education status
        in_education = np.array([age < 18 + S[x] for x in education])

        # Defining unemployed as neither employed or in education
        unemployed = ~employed & ~in_education

        # Assigning employment status each period
        employed[unemployed & (rng.random(N) < lambda1)] = True

        # Assigning unemployment status each period
        employed[employed & (rng.random(N) < sigma)] = False

        # Redefining unemployed as neither employed or in education as changes was made for employed
        employed_group = employed & ~in_education
        unemployed_group = ~employed & ~in_education

        # Illness is randomly assigned
        if include_illness:
            illness_probability = 0.001 * np.exp(0.09 * (age - 18))
            illness = rng.random(N) < illness_probability
        else:
            illness = np.zeros(N, dtype=bool)

        # Illness causes unemployment for 1 period
        employed[illness] = False

        # Redefining unemployed as neither employed or in education as changes was made for employed
        employed_group = employed & ~in_education
        unemployed_group = ~employed & ~in_education

        # Adding the human capital shock
        psi = np.exp(rng.normal(-0.5 * sigma_psi**2, sigma_psi, N))

        # Saving the value of human capital at the beginning of each age as it is needed in the equation for the evolution of human capital
        h_old = h.copy()

        # Incorporates the equations of the evolution of human capital from the exercise
        h[employed_group] = (h_old[employed_group] * (1 + delta_growth[employed_group]) * psi[employed_group])
        h[unemployed_group] = (h_old[unemployed_group] * (1 - delta) * psi[unemployed_group])

        # Illness reduces human capital the period where the agent became ill
        h[illness] *= 0.90

        # Income depending on employment and education
        y = np.zeros(N)
        y[in_education] = y_SU
        y[employed] = h[employed]
        y[unemployed_group] = np.maximum(rho * previous_income[unemployed_group], y_subscribt)

        # Save previous income
        previous_income[employed] = y[employed]

        # Save income
        income_over_time.append(y.copy())

    return np.array(income_over_time)
# ---------------------------------------------------------------------------------------------- #

# Run the baseline model and the model including an illness probability
income_baseline = simulate_illness_model(False)
income_illness = simulate_illness_model(True)

# Calculates the mean income for every age group
mean_baseline = income_baseline.mean(axis=1)
mean_illness = income_illness.mean(axis=1)

# Calculates the Gini coefficient for every age group
gini_baseline_by_age = [gini(x) for x in income_baseline]
gini_illness_by_age = [gini(x) for x in income_illness]

# Calculates the age-dependant illness probability for every age group
illness_probability_by_age = (0.001 * np.exp(0.09 * (np.array(ages) - 18)))

#Plots the mean income for both models
def plot_illness_income():
    plt.figure(figsize=(8, 4))
    plt.plot(ages, mean_baseline, label="Baseline")
    plt.plot(ages, mean_illness, label="With severe illness")
    plt.xlabel("Age")
    plt.ylabel("Mean income")
    plt.title("Effect of Severe Illness on Mean Income")
    plt.legend()
    plt.grid(True)
    plt.show()

#Plots the Gini coefficients for both models
def plot_illness_gini():
    plt.figure(figsize=(8, 4))
    plt.plot(ages, gini_baseline_by_age, label="Baseline")
    plt.plot(ages, gini_illness_by_age, label="With severe illness")
    plt.xlabel("Age")
    plt.ylabel("Gini coefficient")
    plt.title("Effect of Severe Illness on Income Inequality")
    plt.legend()
    plt.grid(True)
    plt.show()

#Plots the age-dependant illness probability
def plot_illness_probability():
    plt.figure(figsize=(8, 4))
    plt.plot(ages, illness_probability_by_age * 100)
    plt.xlabel("Age")
    plt.ylabel("Probability of severe illness (%)")
    plt.title("Probability of Severe Illness by Age")
    plt.grid(True)
    plt.show()