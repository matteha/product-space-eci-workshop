import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
from lifelines import CoxPHFitter


def calculate_MAICS_weights(a, X):
    """
    Calculates the weights using the MAICS method.

    Args:
        a (numpy.ndarray): The coefficient array.
        X (numpy.ndarray): The input array.

    Returns:
        float: The sum of the exponential function applied to the dot product of X and a.
    """
    return np.sum(np.exp(np.matmul(X, a)))


def calculate_MAICS_gradients(a, X):
    """
    Calculates the gradients for the MAICS weights.

    Args:
        a (numpy.ndarray): The coefficient array.
        X (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The dot product of the exponential function applied to the dot product of X and a with X.
    """
    return np.dot(np.exp(np.matmul(X, a)), X)


# hypothetical 'IPD' from IPUMS data: Predict time between graduation and first job (outcome) based on
# 2 treatments: 1) company visit before graduation, 2) networking event before graduation

# set seed for reproducibility
random.seed(42)

# define N of observations
N = 3000

# generate corresponding random data for each column
# NOTE: if not log transforming wealth_parents,
# NOTE: trouble converging in optimization (minimize)
data = {
    "age": [random.randint(30, 45) for _ in range(N)],
    "female": [random.choice([0, 1]) for _ in range(N)],
    "wealth_parents": np.log([random.randint(5000, 55000) for _ in range(N)]),
    "time_to_first_job_in_months": [random.randint(0, 12) for _ in range(N)],
    "event": [random.choice([0, 1]) for _ in range(N)],
}

# create a DataFrame from the data
df = pd.DataFrame(data)

# inspect
df.sample(n=10)

# generate aggregated data (older age, wealthier parents)
import pandas as pd
import numpy as np
import random

# generate random data for each column
data = {
    "age": [random.randint(30, 55) for _ in range(400)],
    "female": [random.choice([0, 1]) for _ in range(400)],
    "wealth_parents": np.log([random.randint(5000, 85000) for _ in range(400)]),
}

# calculate mean of columns
mean_data = {
    "age": np.mean(data["age"]),
    "female": np.mean(data["female"]),
    "wealth_parents": np.mean(data["wealth_parents"]),
}

# Create the dataframe with one row
df_aggregated = pd.DataFrame(mean_data, index=["None"])
df_aggregated.reset_index(inplace=True, drop=True)

# show baseline descriptives, drop missings
variables_maic = ["age", "female", "wealth_parents"]
mean_values = (
    df[variables_maic]
    .dropna()
    .mean()
    .apply(lambda x: round(x, 2) if isinstance(x, float) else x)
)
print(mean_values)

# create new dataframe with centered variables
df_c = pd.DataFrame(
    df[variables_maic].values - df_aggregated[variables_maic].values,
    columns=variables_maic,
)

# inspect
df_c.sample(n=20)

initial_parameters = np.zeros(len(variables_maic))

X = df_c[variables_maic].values

# using scipy
res = minimize(
    fun=calculate_MAICS_weights,
    x0=initial_parameters,
    args=(df_c[variables_maic].values),
    method="BFGS",
    jac=calculate_MAICS_gradients,
)
if res["success"] == False:
    print("not converged, stop")
    STOP

# create weights
wt = np.exp(df_c[variables_maic].dot(res.x))

# copy main df and add weights
df_w = df.copy()
df_w["wt"] = wt

# check if weighted averages match with aggregated data
df_weighted = pd.DataFrame(
    {
        "age": np.average(df_w["age"], weights=df_w["wt"]),
        "female": np.average(df_w["female"], weights=df_w["wt"]),
        "wealth_parents": np.average(df_w["wealth_parents"], weights=df_w["wt"]),
    },
    index=["None"],
)

# -- same format
mean_values_weighted = (
    df_weighted.dropna()
    .mean()
    .apply(lambda x: round(x, 2) if isinstance(x, float) else x)
)
df_aggregated = (
    df_aggregated.dropna()
    .mean()
    .apply(lambda x: round(x, 2) if isinstance(x, float) else x)
)

# -- show
print(mean_values)
print("---------")
print(mean_values_weighted)
print("---------")
print(df_aggregated)

# stop if weighted averages != aggregates
for xvar in mean_values_weighted.index:
    if mean_values_weighted[xvar] != df_aggregated[xvar]:
        print("stop, weighted averages do not match aggregated data")
        STOP

# print effective sample size
ESS = np.sum(wt) ** 2 / np.sum(wt**2)
print("ESS:", ESS)

# stop if effective sample size very small or much smaller than original
# NOTE: somewhat arbitrary numbers here - check literature for recommended values
if ESS < 200 or ESS < (0.3 * df_w.shape[0]):
    print("stop, small effective sample size")
    STOP

# inspect rescaled weights (approaching normal dist. better)
wt_rs = (wt / sum(wt)) * len(df_w)
# -- plot
plt.hist(wt_rs, bins=range(int(min(wt_rs)), int(max(wt_rs)) + 1, 1), edgecolor="black")
plt.show()

# compare hazard ratio outcomes, time to find job after graduation, before / after
# incorporate any weighted measure stat approach here (e.g. weighted cox, weighted t etc.)

# unadjusted Cox proportional hazards model
cox_unadjusted = CoxPHFitter()
cox_unadjusted.fit(df, duration_col="time_to_first_job_in_months", event_col="event")

# adjusted Cox proportional hazards model
cox_adjusted = CoxPHFitter()
cox_adjusted.fit(
    df_w,
    duration_col="time_to_first_job_in_months",
    event_col="event",
    weights_col="wt",
)

# show summaries
cox_unadjusted.print_summary()
cox_adjusted.print_summary()

# plot both
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cox_unadjusted.plot(ax=axes[0])
cox_adjusted.plot(ax=axes[1])
