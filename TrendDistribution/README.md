
# Trends Class

## Overview
The `Trends` class is a tool designed to analyze and visualize trends in time series data. It provides functionality for identifying trends based on a specified threshold, fitting probability distributions to the trends, and plotting various statistical charts.

## Prerequisites
The following Python libraries are required:
- `pandas`
- `matplotlib`
- `numpy`
- `scipy`

Additionally, warnings are suppressed to prevent unnecessary output during execution.

## Class Initialization
The `Trends` class is initialized with the following parameters:
- `history`: A pandas Series representing the time series data, with a `DatetimeIndex`.
- `threshold`: A float value used to determine significant trends based on the relative return.
- `timeperiod`: The frequency for resampling the data (default is `'W'` for weekly).

### Example
```python
trends_instance = Trends(history, threshold=0.05, timeperiod='W')
```

## Methods

### 1. `fit()`
Identifies and characterizes trends within the time series data. It calculates metrics such as:
- `First` and `Last` dates of the trend.
- `Return`: The relative return over the trend.
- `Length`: Duration of the trend in days.
- `Way`: Indicates whether the trend is "Up" or "Down".

The method consolidates adjacent trends that share the same direction, ensuring alternating "Up" and "Down" trends.

#### Returns
The method returns the instance itself with the `trend` attribute populated.

### 2. `get_proba(metric)`
Calculates the probability of continuation for the last identified trend based on the specified `metric` (`Length` or `Return`). It fits either a Pareto or normal distribution to the data and performs a Kolmogorov-Smirnov (KS) test to evaluate the fit.

#### Output
Prints a dictionary with details about the trend, distribution parameters, KS test results, and the continuation probability.

### 3. `get_trend()`
Returns the DataFrame of identified trends.

### 4. `plot_trend()`
Plots the time series data with the identified trends superimposed. The trends are represented as linear segments connecting the starting and ending values.

### 5. `plot_distrib(metric)`
Plots the histogram of the `Length` and overlays the fitted probability Pareto distribution. The plot also marks the last trend's value with a dashed vertical line.

### 6. `plot_cumdistrib(metric)`
Plots the cumulative distribution function (CDF) and the complementary CDF (RCDF) for the `Length` or `Return`. It fits the appropriate distribution and overlays the cumulative probability values.

## Usage Example
```python
import pandas as pd
import numpy as np
from trendist_toolkit import trends

# Generate sample data
dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
values = np.random.normal(100, 10, len(dates))
history = pd.Series(values, index=dates, name='Sample Data')

# Create Trends instance
trends_instance = trends(history, threshold=0.05, timeperiod='W')

# Fit the trends
trends_instance.fit()

# Get probability of continuation
trends_instance.get_proba()

# Plot the trends
trends_instance.plot_trend()

# Plot the distribution of Length
trends_instance.plot_distrib()

# Plot the cumulative distribution of Return
trends_instance.plot_cumdistrib()
```

## Notes
- The class expects the `history` parameter to be a pandas Series with a `DatetimeIndex`.
- The trend consolidation ensures an alternating pattern of "Up" and "Down" trends for better visualization and analysis.
- For accurate fitting and probability estimation, ensure the data does not contain significant outliers or missing values.
