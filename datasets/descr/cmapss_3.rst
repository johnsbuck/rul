C-MAPSS 3 Data Set
==================

Notes
-----
Data Set Characteristics:

    :Number of Features: 26

    :Feature Information (in order):
        - Unit number
        - Cycle number
        - Operating Settings (1-3)
        - Sensors (1-21)

    :Fault Modes: 2

    :Operating Conditions: 1

    Training Set:

        :Number of Engine Units: 100

        :Number of Data Points: 24720

    Testing Set:

        :Number of Engine Units: 100

        :Number of Data Points: 16596

    :Creator: Saxena, A. and Goebel, K.


This is a copy of the 1st Commercial Aero-Propulsion System Simulations Turbofan Engine Data Set.
https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan

This dataset was taken from the NASA Prognostics Center of Excellence (PCoE) website and is provided by the
Prognostics CoE at NASA Ames Research Center.

Experimental Scenario:

    *(Excerpt of the PHM08 Prognostics Data Challenge Data Set Description from the PHM08 Challenge Data Set)*

    Data sets consist of multiple multivariate time series. Each data set is further divided into training and test
    subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of
    engines of the same type. Each engine starts with different degrees of initial wear and manufacturing
    variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not
    considered a fault condition. There are three operational settings that have a substantial effect on engine
    performance. These settings are also included in the data. The data are contaminated with sensor noise.

    The engine is operating normally at the start of each time series, and starts to degrade at some point
    during the series. In the training set, the degradation grows in magnitude until a predefined threshold is
    reached beyond which it is not preferable to operate the engine. In the test set, the time series ends some
    time prior to complete degradation. The objective of the competition is to predict the number of
    remaining operational cycles before in the test set, i.e., the number of operational cycles after the last
    cycle that the engine will continue to operate properly.

Submission:

    This data set contains the correct RUL for each unit in the testing set, allowing the user to do direct comparisons
    between their results and the actual results.

**References**

    - A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA