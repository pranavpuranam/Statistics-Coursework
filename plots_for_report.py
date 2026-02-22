# imports

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.stats as stats

# set the random seed using college id

seed = 2194496
random.seed(seed)
np.random.seed(seed)
rng = np.random.default_rng(seed)

# -----------------------------
# Question 1
# -----------------------------

# Part 1 ----------

"""
The marginal distribution is easily derived from the mean-vector and variance-covariance matrix:

X1 ~ N(mu1, Sigma11) = N(0, 2)

The conditional distribution is derived using the formula from p. 77 of the statistics course notes:

X2 | X1 = x1 ~ N(mu2 + sigma2/sigma1 * rho * (x1 - mu1), (1 - rho**2) * sigma1**2) = N(1 - x/2, 7/2)
"""

# Part 2 ----------

mu_true = np.array([0.0, 1.0])
sigma_true = np.array([[2.0, -1.0], [-1.0, 4.0]]) # data from the matrices in the question

n = 100 # number of samples specified in the question

x1 = rng.normal(loc = mu_true[0], scale = np.sqrt(sigma_true[0, 0]), size = n) # normal distribution

# Part 3 ----------

cond_mu = 1 - x1/2
cond_var = 7/2 # identified in Part 1 answer

x2 = rng.normal(loc = cond_mu, scale = np.sqrt(cond_var), size = n) # normal distribution

plt.figure()
plt.scatter(x1, x2, color="#0000cc")
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("bivariate_sample.png", dpi=300, bbox_inches="tight")
plt.show()

# Part 4 ----------

X = np.column_stack([x1, x2])

def unpack(theta): # building the mean-vector and variance-covariance matrix
    mu1, mu2, log_l11, l21, log_l22 = theta
    
    l11 = np.exp(log_l11)
    l22 = np.exp(log_l22) # forces the two variances to be > 0
    
    L = np.array([[l11, 0.0], [l21, l22]])

    Sigma = L @ L.T
    mu = np.array([mu1, mu2])

    return mu, Sigma # outputs results

def neg_loglik(theta, X): #  for a given theta and data we compute the negative log-likelihood of the data
    mu, Sigma = unpack(theta)
    return - np.sum(multivariate_normal.logpdf(X, mean = mu, cov = Sigma)) # minus as we are using a minimizer

m0 = X.mean(axis=0) # starting guess for mu
S0 = np.cov(X, rowvar=False, bias=False) # starting guess for Sigma
L0 = np.linalg.cholesky(S0 + 1e-10*np.eye(2)) # stable cholesky (tiny jitter)

theta0 = np.array([m0[0], m0[1], np.log(L0[0, 0]), L0[1, 0], np.log(L0[1, 1])])   

res = minimize(neg_loglik, theta0, args=(X,), method="L-BFGS-B")  # numerical MLE
mu_hat, Sigma_hat = unpack(res.x)

print(mu_hat[0], mu_hat[1], Sigma_hat[0, 0], Sigma_hat[0, 1], Sigma_hat[1, 1]) # results very similar to original matrices so algorithm is good.

# -----------------------------
# Question 2
# -----------------------------

# Part 1 ----------

N = 10000 # number of samples
m = 50 # observations in each sample 

X = rng.binomial(1, 0.45, size=(N, m)) # generate samples

# Part 2 ----------

alpha = 0.05
z = norm.ppf(1 - alpha/2) # manually computing the z-value corresponding to 95% CI

print(z)

thetahat = X.mean(axis=1) # sample means

den = 1 + (z**2)/m
center = (thetahat + (z**2)/(2*m)) / den
half = (z * np.sqrt(thetahat*(1-thetahat)/m + (z**2)/(4*m**2))) / den # numerical implementation of method 1

L1 = center - half
U1 = center + half # lower and upper bounds

# Part 3 ----------

theta_true = 0.45
pct = 100 * np.mean((L1 <= theta_true) & (theta_true <= U1)) # checking percent that fall within bounds
print("Method 1:", pct)

# Part 4 ----------

se2 = np.sqrt(thetahat * (1 - thetahat) / m)

L2 = thetahat - z * se2
U2 = thetahat + z * se2 # numerical implementation of method 2

L2 = np.clip(L2, 0, 1)
U2 = np.clip(U2, 0, 1)

pct2 = 100 * np.mean((L2 <= theta_true) & (theta_true <= U2))
print("Method 2:", pct2)

# Part 5 ----------

"""
[m = 50] 
Method 1: 95.37%
Method 2: 93.38%

[m = 20] 
Method 1: 96.00%
Method 2: 92.38%

[m = 100]
Method 1: 94.34%
Method 2: 94.34%

Method 2 suffers from small-sample bias; the samplig distribution of thetacap is not very normal and slightly skewed. The interval tends to be too
narrow and poorly centred, leading to lower coverage for m = 20 and m = 50 but less of an issue at m = 100.

Method 1 corrects for small-sample bias with adjusted center and slightly wider intervals. This means it stays closer to 95% for all m, but it does
tend to slightly overcover at m = 20.
"""

# PLOT

# Parameters
N = 10000
theta_true = 0.45
alpha = 0.05
z = norm.ppf(1 - alpha/2)

m_values = np.arange(10, 201, 5)

coverage1 = []
coverage2 = []

for m in m_values:
    X = rng.binomial(1, theta_true, size=(N, m))
    thetahat = X.mean(axis=1)

    # Method 1 (Wilson)
    den = 1 + (z**2)/m
    center = (thetahat + (z**2)/(2*m)) / den
    half = (z * np.sqrt(thetahat*(1-thetahat)/m + (z**2)/(4*m**2))) / den
    L1 = center - half
    U1 = center + half
    cov1 = 100 * np.mean((L1 <= theta_true) & (theta_true <= U1))
    coverage1.append(cov1)

    # Method 2 (Wald)
    se2 = np.sqrt(thetahat * (1 - thetahat) / m)
    L2 = thetahat - z * se2
    U2 = thetahat + z * se2
    L2 = np.clip(L2, 0, 1)
    U2 = np.clip(U2, 0, 1)
    cov2 = 100 * np.mean((L2 <= theta_true) & (theta_true <= U2))
    coverage2.append(cov2)

# Plot
plt.figure()

plt.plot(m_values, coverage1,
         color="red",
         linewidth=2,
         marker="o",
         label="Method 1")

plt.plot(m_values, coverage2,
         color="#0000cc",
         linewidth=2,
         linestyle="--",
         marker="s",
         label="Method 2")

plt.axhline(95,
            color="black",
            linewidth=1.5,
            linestyle=":",
            label="Nominal 95%")

plt.xlabel("Sample Size")
plt.ylabel("Empirical Coverage (%)")

plt.legend()
plt.savefig("sample_coverage.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# Question 3
# -----------------------------

# Part 1 ----------

# (a)

weather = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Statistics-Coursework/pp1422_weather.csv")
pm10 = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Statistics-Coursework/pp1422_PM10.csv") # reading in the raw files

weather["Date"] = pd.to_datetime(weather[["Year", "Month", "Day"]]).dt.strftime("%d/%m/%Y") # add a date column to weather for left joining

weather = weather.drop(columns=["Day", "Month", "Year"])

weather = weather.set_index("Date")
pm10 = pm10.set_index("Date")

build = weather.join(pm10, how = "left") # left join the datasets

build = build.replace("-", np.nan)
build = build.dropna() # select days for which both pollution and weather data are available

build.to_csv("build.csv")

# (b)

cols = ["Wind", "Cloud", "Temperature", "Humidity", "PM10"]
build[cols] = build[cols].apply(pd.to_numeric, errors="coerce") # parse the cols as numeric

for col in cols:
    plt.figure()
    plt.hist(build[col], bins=20)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {col}")
    plt.show()

summary = build[cols].agg(["mean", "median", "std", "min", "max"])
print(summary.head(10))

cols = ["Wind", "Cloud", "Temperature", "Humidity", "PM10"]
build[cols] = build[cols].apply(pd.to_numeric, errors="coerce")

for col in cols:
    plt.figure()
    plt.hist(build[col].dropna(), bins=20, 
             color="#0000cc")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    # Save only the PM10 histogram
    if col == "PM10":
        plt.savefig("PM10_histogram.png", dpi=300, bbox_inches="tight")
    
    plt.show()

summary = build[cols].agg(["mean", "median", "std", "min", "max"])
print(summary)

# (c)

print("pm10 '-' count:", (pm10 == "-").sum().sum()) # missing values in PM10 data

Q1 = build[cols].quantile(0.25)
Q3 = build[cols].quantile(0.75)
IQR = Q3 - Q1 # defining interquartile range

outliers = ((build[cols] < (Q1 - 1.5*IQR)) | (build[cols] > (Q3 + 1.5*IQR))).sum() # counting the number of outliers outside 1.5 * IQR

print("Outliers:")
print(outliers)

"""
The weather dataset contains 121 rows of data, without any missing values. The PM10 dataset contains 366 rows of data, including 5 missing values 
denoted by '-'. After merging and removing missing values, the built dataset contains 118 rows of data.

Using a 1.5*IQR test on the merged dataset, we find 6 outliers in Wind, 4 in Cloud, 7 in Temperature, 0 in Humidity, and 3 in PM10.
"""

# Part 2 ----------

# (a)

y = build["PM10"].to_numpy() # target variable

predictors = ["Wind", "Cloud", "Temperature", "Humidity"] # we are modelling PM10 as a function of these 4 variables

rows = []
for xname in predictors:
    x = build[xname].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat # calculate residuals
    sse = np.sum(resid**2)
    mse = sse / len(y)
    s2 = sse / (len(y) - 2)
    r2 = 1 - sse / np.sum((y - y.mean())**2)
    rows.append([xname, beta[0], beta[1], r2, sse, mse, s2]) # add in regression parameters and error metrics

results = pd.DataFrame(rows, columns=["x", "alpha_hat", "beta_hat", "r2", "sse", "mse", "s2"])
results = results.sort_values(["r2", "s2"], ascending=[False, True]) # ramk by highest r2 and s2

# Multiple linear regression: PM10 ~ Wind + Cloud + Temperature + Humidity

y = build["PM10"].to_numpy()
X_multi = build[["Wind", "Cloud", "Temperature", "Humidity"]].to_numpy()

# Add intercept
X_multi = np.column_stack([np.ones(len(X_multi)), X_multi])

# Least squares estimate
beta_multi = np.linalg.lstsq(X_multi, y, rcond=None)[0]

yhat_multi = X_multi @ beta_multi
resid_multi = y - yhat_multi

sse_multi = np.sum(resid_multi**2)
s2_multi = sse_multi / (len(y) - X_multi.shape[1])  # n - p
r2_multi = 1 - sse_multi / np.sum((y - y.mean())**2)

print("Multiple regression coefficients:")
print("alpha_hat =", beta_multi[0])
print("beta_hats =", beta_multi[1:])
print("R^2 =", r2_multi)
print("Residual variance (s^2) =", s2_multi)

print(results) # print every model with best ones at the top and worst at the bottom

"""
The model which explains PM10 concentraction as a function of Wind is the best, with the highest R-squared and lowest s2 value.
"""

# (b)

best_row = results.iloc[0]
xname, alpha_hat, beta_hat = best_row["x"], best_row["alpha_hat"], best_row["beta_hat"]

x = build[xname].to_numpy()
y = build["PM10"].to_numpy()

x_line = np.linspace(x.min(), x.max(), 100)
y_line = alpha_hat + beta_hat * x_line

plt.figure()
plt.scatter(x, y, color="#0000cc")
plt.plot(x_line, y_line, color="red")
plt.xlabel(xname)
plt.ylabel("PM10")

plt.savefig("pm10_wind.png", dpi=300, bbox_inches="tight")
plt.show()

"""
The model appears reasonably apropriate but not perfect. There is a clear negative linear association between Wind and PM10, as wind increases, PM10
generally decreases, supporting a linear fitting model.

However:
(1) there is substantial scatter around the line
(2) there are clear outliers, such as the high PM10 value around Wind ~= 1-1.5
(3) spread looks larger at low wind than high wind, indicating possible mild heteroscedasticity
"""

# (c)

x = build[xname].to_numpy()
y = build["PM10"].to_numpy()

y_hat = alpha_hat + beta_hat * x
residuals = y - y_hat

plt.figure()
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

# Scatter points (blue)
plt.scatter(osm, osr, color="#0000cc")

# Fitted QQ line (red)
plt.plot(osm, slope * osm + intercept, color="red")

plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

plt.savefig("qq_residuals.png", dpi=300, bbox_inches="tight")
plt.show()

"""
The center lies reasonably close to the line, indicating residuals are roughly normal. However, the right tail bends strongly upward, with the largest
residuals being much larger than normal would predict. The left tail also deviates slightly.

Mainly due to the heavy right tail, the assumption of normality is questionable.
"""