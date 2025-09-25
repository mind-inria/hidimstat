.. _developer_guidelines:

Developer Guidelines
====================

This section provides guidelines for developers contributing to the hidimstat project. 

.. contents:: Table of Contents
   :depth: 2
   :local:

.. sectnum::
   :depth: 2

Reproducibility and Randomness
------------------------------

Design choice: `np.random.Generator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hidimstat uses numpy.random.Generator for all randomness because it provides a reliable 
way to manage separate, reproducible streams of random numbers. The key benefit is the 
ability to spawn new, independent Generator instances from a single parent.

This is particularly useful for our methods that employ parallel samplers or other 
algorithms requiring multiple, distinct random streams. Spawning ensures that each 
parallel process has its own unique sequence of random numbers, preventing interference 
and guaranteeing that the overall result remains reproducible.

The Generator object for any given method is instantiated by the internal 
`check_random_state` utility function, which handles the logic based on the user's 
input.


The random_state Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~

The behavior of a method's randomness is exclusively controlled by the random_state 
parameter. It accepts three types of input:

* `int`
  An integer input ensures both **reproducibility**: running the script twice 
  produces identical results, and **repeatability**: calling the method twice within the 
  same script, such as `<method>.importance()`, yields the exact same output.
* `numpy.random.Generator`
  Passing a Generator object provides reproducibility but not 
  repeatability. The stream of random numbers is reproducible if the user seeds their 
  Generator object, but subsequent calls to the method will advance the generator's 
  state, leading to different results.
* `None`
  This input provides neither reproducibility nor repeatability. The random 
  numbers are generated from a non-deterministic source, resulting in different outcomes 
  on each run and each call.


Testing
~~~~~~~
All methods that involve randomness must include a dedicated test suite. The following 
test patterns help to ensure consistent behavior:

* `test_<method>_repeatability`
* `test_<method>_randomness_with_none`
* `test_<method>_reproducibility_with_integer`
* `test_<method>_reproducibility_with_rng`
