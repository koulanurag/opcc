Offline Policy Comparison with Confidence : Benchmarks and Baselines
====================================================================

.. admonition:: TL;DR
   :class: note

    It's a benchmark comprising **"policy comparison queries"(pcq)** to evaluate uncertainty estimation in offline
    reinforcement learning.


----------
Abstract
----------
Decision makers often wish to use offline historical data to compare sequential-action policies at various world states. Importantly, computational tools should produce confidence values for such offline policy comparison (OPC) to account for statistical variance and limited data coverage. Nevertheless, there is little work that directly evaluates the quality of confidence values for OPC. In this work, we address this issue by creating benchmarks for OPC with Confidence (OPCC), derived by adding sets of policy comparison queries to datasets from offline reinforcement learning. In addition, we present an empirical evaluation of the risk versus coverage trade-off for a class of model-based baselines. In particular, the baselines learn ensembles of dynamics models, which are used in various ways to produce simulations for answering queries with confidence values. While our results suggest advantages for certain baseline variations, there appears to be significant room for improvement in future work.

----------
Slides
----------
.. raw:: html

    <div class="google-slides-container">
    <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRtwZFzF081XdGn8jFUIXkdMrRmtQMLIOqgw1ivh6a554KwmtlnWi9zWfjw1fh5WsSQMEiVU0s8RyzN/embed?start=false&loop=false&delayms=3000" frameborder="0" width="480" height="299" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
    </div>

----------
Contents
----------
.. toctree::
   :maxdepth: 2

   installation
   quick-start
   benchmark-information
   development
   api


----------------
Contact
----------------

If you have any questions or suggestions , please open an issue on this GitHub repository.

-------------------
Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
