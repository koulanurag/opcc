==========================================================================
mujoco(gym)
==========================================================================

`Reference <https://gym.openai.com/envs/#mujoco>`_

.. raw:: html

    <p float="left">
        <img width="160" alt="mujoco-halfcheetah" src="_static/images/halfcheetah.png" />
        <img width="160" alt="mujoco-hopper" src="_static/images/hopper.png" />
        <img width="160" alt="mujoco-walker2d" src="_static/images/walker2d.png" />
    </p>


Datasets
--------


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Environment Name
     - Datasets
     - Query-Count
   * - `HalfCheetah-v2`
     - `random, expert, medium, medium-replay, medium-expert`
     - `1500`
   * - `Hopper-v2`
     - `random, expert, medium, medium-replay, medium-expert`
     - `1500`
   * - `Walker2d-v2`
     - `random, expert, medium, medium-replay, medium-expert`
     - `1500`


Pre-trained policy performance
------------------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Environment Name
     - `pre_trained=1` (best)
     - `pre_trained=2`
     - `pre_trained=3`
     - `pre_trained=4` (worst)

   * - `HalfCheetah-v2`
     - 1169.13±80.45
     - 1044.39±112.61
     - 785.88±303.59
     - 94.79±40.88

   * - `Hopper-v2`
     - 1995.84±794.71
     - 1466.71±497.1
     - 1832.43±560.86
     - 236.51±1.09

   * - `Walker2d-v2`
     - 2506.9±689.45
     - 811.28±321.66
     - 387.01±42.82
     - 162.7±102.14

