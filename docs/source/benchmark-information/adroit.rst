==========================================================================
Adroit
==========================================================================

`Reference <https://github.com/Farama-Foundation/D4RL/wiki/Tasks#adroit>`_

.. raw:: html

    <p float="left">
        <img width="160" alt="pen-v0" src="_static/images/pen.png" />
        <img width="160" alt="door-v0" src="_static/images//door.png" />
        <img width="160" alt="door-v0" src="_static/images//hammer.png" />
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
   * - `d4rl:pen-v0`
     - `human, cloned, expert`
     - `1500`
   * - `d4rl:door-v0`
     - `human, cloned, expert`
     - `1500`
   * - `d4rl:hammer-v0`
     - `human, cloned, expert`
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

   * - `d4rl:pen-v0`
     - 1169.13±80.45
     - 1044.39±112.61
     - 785.88±303.59
     - 94.79±40.88

   * - `d4rl:door-v0`
     - 1995.84±794.71
     - 1466.71±497.1
     - 1832.43±560.86
     - 236.51±1.09

   * - `d4rl:door-v0`
     - 14863.43±3592.63
     - 7057.41±7514.68
     - 665.99±3454.75
     - -231.54±79.61
