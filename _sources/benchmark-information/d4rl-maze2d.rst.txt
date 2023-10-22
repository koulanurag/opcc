==========================================================================
d4rl:maze2d
==========================================================================

`Reference <https://github.com/rail-berkeley/d4rl/wiki/Tasks#maze2d>`_

.. image:: https://github.com/rail-berkeley/offline_rl/raw/assets/assets/mazes_filmstrip.png
  :width: 500
  :alt: maze2d-environments


Datasets
--------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Environment Name
     - Datasets
     - Query-Count
   * - `d4rl:maze2d-open-v0`
     - `1k, 10k, 100k, 1m`
     - `1500`
   * - `d4rl:maze2d-medium-v1`
     - `1k, 10k, 100k, 1m`
     - `1500`
   * - `d4rl:maze2d-umaze-v1`
     - `1k, 10k, 100k, 1m`
     - `1500`
   * - `d4rl:maze2d-large-v1`
     - `1k, 10k, 100k, 1m`
     - `121`


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

   * - `d4rl:maze2d-open-v0`
     - 122.2±10.61
     - 104.9±22.19
     - 18.05±14.85
     - 4.85±8.62

   * - `d4rl:maze2d-medium-v1`
     - 245.55±272.75
     - 203.75±252.61
     - 256.65±260.16
     - 258.55±262.81

   * - `d4rl:maze2d-umaze-v1`
     - 235.5±35.45
     - 197.75±58.21
     - 23.4±73.24
     - 3.2±9.65

   * - `d4rl:maze2d-large-v1`
     - 231.35±268.37
     - 160.8±201.97
     - 50.65±76.94
     - 9.95±9.95
