# Sustav za Predviđanje Dionica

# Why do we need regression?

This is because linear regression is the most basic kind of prediction. In most of data science filed, what we want to do is to figure out what’s going on the data, and to **predict what will happen in the future**. In finance, for example, we could find that the stock prices of two companies are close to linear (the example below). If the pattern has lasted for long, we can expect the linear patten will remain in the future. This leads to a prediction of stock price.



The example data we’re going to analyze is relative performance of the sector “Computer and Technology” to the sector “Business Services”. The graph below shows their relative performance, and you can see that it’s close to linear. If the graph is close to a clear line, it means the performances of the two sectors are strongly correlated. If the slope of the line is large, it means the performance of “Computer and Technology” is better than “Business Services”.

## 1. Import packages

As we do in other stories, we import “numpy”, “matplotlib”, and “pandas” for basic data analysis. “datetime” is a must when dealing with time series data. Because we have to make regression, we need “sklearn” as well. This does every math things for you.



## 4. Prepare data

Before applying linear regression, we have to convert input data into a form suitable for “sklearn”. In the code below, the data for the x-axis is denoted as “X”, while the data for the y-axis “y”. “X” is made from the datetime objects we made earlier.

The next step is the most important one of this story. Because we can’t feed datetime objects directly, we must convert them into float values. The function “financialanalysis” converts each date into a float year. Float year means each data is represented in year. For example, 2020–07–01 becomes 2020.49 because is middle of the year.

The operation “[::, None]” converts a row array into a column array. We can’t feed row arrays.

![img](https://miro.medium.com/max/700/1*_cTNlqK11KDCGwUIDoKpug.png)

## 5. Apply linear regression

Finally, we use the function “LinearRegression().fit()” of sklearn to apply linear regression on “X” and “y”. The returned object “reg” contains the slope and y-intercept of the prediction line. Once we extract the slope and intercept, we generate the line with “slope*X + intercept”. But we have to note here is that, because X is a column array, “fittedline” is also a column vector. Thus, we make it back to a row vector with the “flatten()” function.



## 6. Make graphs

Then we make the graph of the original data and the prediction line. If you don’t know how to use Matplotlib, the following article explains the basics:
