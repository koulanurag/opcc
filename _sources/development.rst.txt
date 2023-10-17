=============
Development
=============

We begin by installing following dependencies:

    .. code-block:: console

        pip install -e ".[dev]"

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://github.com/koulanurag/opcc/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/koulanurag/opcc/actions/workflows/python-package.yml
   :alt: Python package
.. image:: https://img.shields.io/github/license/koulanurag/opcc
   :target: https://img.shields.io/github/license/koulanurag/opcc
   :alt: License
.. image:: https://codecov.io/gh/koulanurag/opcc/branch/main/graph/badge.svg?token=47LIB1CLI4
   :target: https://codecov.io/gh/koulanurag/opcc
   :alt: codecov

------------------
Training Policies
------------------

We primarily use :code:`td3` for training policies and hand-pick checkpoints at regular intervals to get policies of various qualities.

Run (example):

.. code-block:: console

    python scripts/td3.py --env Hopper-v2

For more details, refer to:

.. code-block:: console

    python scripts/td3.py --help

The policies are saved in :code:`opcc/assets/<env-name>/model/model_<model-id>.p`, where :code:`id` is a whole number. For semantic reasons, we assign larger number to poor quality policies.

-----------------
Generate-Queries
-----------------
In order to generate queries for the considered environments and selected policies, we run following commands

.. code-block:: console

    % following commands were used to generate queries
    python scripts/generate_queries.py --env-name HalfCheetah-v2 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.1 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name Hopper-v2 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.1 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name Walker2d-v2 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.1 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name d4rl:maze2d-large-v1 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.2 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name d4rl:maze2d-umaze-v1 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.2 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name d4rl:maze2d-medium-v1 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.2 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb
    python scripts/generate_queries.py --env-name d4rl:maze2d-open-v0 --horizons 10 20 30 40 50 --policy-ids 1 2 3 4 --noise 0.5 --eval-runs 10 --ignore-delta 10 --max-trans-count 2000 --ignore-stuck-count 1000 --save-prob 0.6 --per-policy-comb-query 250 --use-wandb

You can understand available command attributes using following command

.. code-block:: console

    python scripts/generate_queries.py --help
--------------------------------------------------------
Pre-trained policy stats
--------------------------------------------------------

Output of following command is used to updated benchmark information in readme.md or docs/source/benchmark-information.rst

.. code-block:: console

    python scripts/generate_policy_stats.py --all-envs

Also, refer to following for more usage details:

.. code-block:: console

    python scripts/generate_policy_stats.py --help


----------------
Testing Package
----------------

#. Install Dependencies :
    .. code-block:: console

        pip install -e ".[test]"

#. Testing is computationally expensive as we validate ground truth value estimates and corresponding labels. These can be disabled by setting following flags:

    .. code-block:: console

       export SKIP_QUERY_TARGET_TESTS=1 # disable target estimation and label validation
       export SKIP_Q_LEARNING_DATASET_TEST=1  # disable test for checking dataset existence
       export SKIP_SEQUENCE_DATASET_TEST=1 # disables test for checking sequence dataset

#. Run:
    .. code-block:: console

        pytest -v --xdoc


----------------
Generate Docs
----------------

#. Install dependencies

    .. code-block:: console

        pip install -e ".[docs]"


#. Generate Sphinx Doc

    .. code-block:: console

        sphinx-build -M html docs/source/ docs/build/ -a
