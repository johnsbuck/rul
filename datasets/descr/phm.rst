Prognostics Heatlh Management 2008 Challenge Data Set
=====================================================

Notes
-----
Data Set Characteristics:

    :Number of Features: 26

    :Feature Information (in order):
        - Unit number
        - Cycle number
        - Operating Settings (1-3)
        - Sensors (1-21)

    :Fault Modes: 1

    :Operating Conditions: 6

    Training Set:

        :Number of Engine Units: 218

        :Number of Data Points: 45918

    Testing Set:

        :Number of Engine Units: 218

        :Number of Data Points: 29820

    Final Submission Set:

        :Number of Engine Units: 435

        :Number of Data Points: 55156

    :Creator: Saxena, A. and Goebel, K.


This is a copy of the Prognostics Health Management 2008 (PHM08) Challenge Turbofan Engine Data Set.
https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#phm08_challenge

This dataset was taken from the NASA Prognostics Center of Excellence (PCoE) website and is provided by the
Prognostics CoE at NASA Ames Research Center.

The PHM08 dataset was originally used in the 1st international Prognostics and Health Management conference held in
2008, and was used for a data challenge available to the attendees. The original challenge had three winners whose
papers are in references.

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

Evaluation:

    *(Excerpt of the PHM08 Prognostics Data Challenge Data Set Description from the PHM08 Challenge Data Set)*

    The final score is a weighted sum of RUL errors. The scoring function is an asymmetric function that penalizes late
    predictions more than the early predictions. Following equations describe this function analytically.

    ![Score Plot](score_plot.png)

    ![Score Function](score.png)

Submission:

    The first testing set results (218 RULs) can be submitted once a day at:
    https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#phm08_challenge

    The second testing set, also described as the final submission set, was only submittable once in the original
    data challenge and the current submission page is unknown. It is suggested not to use this set as you are unable
    to score your answer, and most papers have been referring to the first testing set score as a result.

**References**

    - A. Saxena and K. Goebel (2008). "PHM08 Challenge Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
    - Heimes, F.O., “Recurrent neural networks for remaining useful life estimation”, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
    - Tianyi Wang, Jianbo Yu, Siegel, D., Lee, J., “A similarity-based prognostics approach for Remaining Useful Life estimation of engineered systems”, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
    - Peel, L., “Recurrent neural networks for remaining useful life estimation”, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
