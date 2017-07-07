# Python Filters

A major component in improving Remaining Useful Life estimations 
is the use of filters to either pre-process data before inputting into a model or
post-processing after receiving the output of said model. A commonly used filter is
a basic Kalman filter used for post-processing in the original PHM 2008 Competition, 
which was improved upon later with the use of a Switching Kalman Filter.

Improvements can also be seen using a Butterworth filter or Savitzky-Golay filter, 
making these different filters give us a different view into our data that can be
effective. Because of this, a filter module is needed.

## Planned Development

[FilterPy](https://github.com/rlabbe/filterpy) is a educational Python filter module
that accompanies the creator's book,
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/),
which focuses on basic filters and Kalman filters. This filter isn't optimized, as it
is meant for educational purposes. With this module, we hope to focus on the following
filters:

* Kalman Filters
    * Basic
    * Switching
* Savitzky-Golay
* Butterworth

Other filters that have been covered by the FilterPy might be covered as well, 
but will **not** include incredibly basic filters such as the *g & h filter*.

## Separation

It is possible for this module to be separated from the rest of this project as
its own package. Any such planning will be done in the later phases after implementing
the 4 filters previously discussed.