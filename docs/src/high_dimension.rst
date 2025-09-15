.. _high_dimension:


===========================
Inference in high dimension
===========================

Naive inference in high dimension is ill-posed
----------------------------------------------

In some cases, data represent high-dimensional measurements of some phenomenon of interest (e.g. imaging or genotyping). The common characteristic of these problems is to be very high-dimensional and lead to correlated features. Both aspects are clearly detrimental to conditional inference, making it both expensive and powerless:

* Expensive: most learers are quadratic or cubic in the number of features. Moreover per-feature inference generally entails a loop over features
* powerless: As dimensionality and correlation increase, it  becomes harder and harder to isolate the contribution of each variable, meaning that conditional inference is ill-posed.

This is illustrated in the above example, where the Desparsified Lasso struggles
to identify relevant features



.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`https://hidimstat.github.io/dev/generated/gallery/examples/plot_2D_simulation_example.html#`
