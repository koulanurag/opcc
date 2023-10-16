============
Quick Start
============

In the section, we understand minimal and sufficient usage of  `opcc` framework.

---------
Queries
---------

Following code is a demo of retrieving queries and iterating over them.

.. code-block:: python
    :linenos:

    import opcc
    import numpy as np
    from sklearn import metrics

    env_name = 'HalfCheetah-v2'
    dataset_name = 'random'

    # ########################################################
    # Policy Comparison Queries (PCQ) (Section : 3.1 in paper)
    # ########################################################
    # Queries are dictionaries with policies as keys and
    # corresponding queries as values.
    queries = opcc.get_queries(env_name)


    # ########################################################
    # Following is a random answer predictor for a query
    # ########################################################
    def random_predictor(obs_a, obs_b, action_a, action_b,
                         policy_a, policy_b, horizon):
        answer = np.random.randint(low=0, high=2, size=len(obs_a)).tolist()
        confidence = np.random.rand(len(obs_a)).tolist()
        return answer, confidence


    # ########################################################
    # Query Iterator
    # ########################################################
    targets = []
    predictions = []
    confidences = []
    # Batch iteration through Queries :
    for (policy_a_id, policy_b_id), query_batch in queries.items():
        # retrieve policies
        policy_a, _ = opcc.get_policy(*policy_a_id)
        policy_b, _ = opcc.get_policy(*policy_b_id)

        # query-a
        obs_a = query_batch['obs_a']
        action_a = query_batch['action_a']

        # query-b
        obs_b = query_batch['obs_b']
        action_b = query_batch['action_b']

        # horizon for policy evaluation
        horizon = query_batch['horizon']

        # ground truth binary vector:
        # (Q(obs_a, action_a, policy_a, horizon)
        # <  Q(obs_b, action_b, policy_b, horizon))
        target = query_batch['target'].tolist()
        targets += target

        # Let's make predictions for the given queries.
        # One can use any mechanism to predict the corresponding
        # answer to queries, and we simply use a random predictor
        # over here for demonstration purposes
        p, c = random_predictor(obs_a, obs_b, action_a, action_b,
                                policy_a, policy_b, horizon)
        predictions += p
        confidences += c



-------------------
Evaluation Metrics
-------------------

.. code-block:: python
    :linenos:
    :emphasize-lines: 24, 27, 33

    # #########################################
    # (Section 3.3 in paper)
    # #########################################
    loss = np.logical_xor(predictions, targets)  # we use 0-1 loss for demo

    # List of tuples (coverage, selective_risks, tau)
    coverage_sr_tau = []
    tau_interval=0.01
    for tau in np.arange(0, 1 + 2 * tau_interval, tau_interval):
      non_abstain_filter = confidences >= tau
      if any(non_abstain_filter):
        selective_risk = np.sum(loss[non_abstain_filter])
        selective_risk /= np.sum(non_abstain_filter)
        coverage = np.mean(non_abstain_filter)
        coverage_sr_tau.append((coverage, selective_risk, tau))
      else:
        # 0 risk for 0 coverage
        coverage_sr_tau.append((0, 0, tau))

    coverages, selective_risks, taus = list(zip(*sorted(coverage_sr_tau)))
    assert selective_risks[0] == 0 and coverages[0] == 0 , "no coverage not found"
    assert coverages[-1] == 1, 'complete coverage not found'

    # AURCC ( Area Under Risk-Coverage Curve): Ideally, we would like it to be 0
    aurcc = metrics.auc(x=coverages,y=selective_risks)

    # Reverse-pair-proportion
    rpp = np.logical_and(np.expand_dims(loss, 1)
                         < np.expand_dims(loss, 1).transpose(),
                         np.expand_dims(confidences, 1)
                         < np.expand_dims(confidences, 1).transpose()).mean()

    # Coverage Resolution (cr_k) : Ideally, we would like it to be 1
    k = 10
    bins = [_ for _ in np.arange(0, 1, 1 / k)]
    cr_k = np.unique(np.digitize(coverages, bins)).size / len(bins)

    print("aurcc: {}, rpp: {}, cr_{}:{}".format(aurcc, rpp, k, cr_k))


---------
Dataset
---------

.. code-block:: python
    :linenos:

    # ###########################################
    # Datasets: (Section 4 in paper - step (1) )
    # ###########################################

    import opcc

    env_name = 'HalfCheetah-v2'

    # list all dataset names corresponding to an env
    dataset_names = opcc.get_dataset_names(env_name)

    dataset_name = 'random'
    # This is a very-slim wrapper over D4RL datasets.
    dataset = opcc.get_qlearning_dataset(env_name, dataset_name)


--------------------
Policy Usage
--------------------

.. code-block:: python
    :linenos:

    import opcc, gym, torch

    env_name = "HalfCheetah-v2"
    policy, policy_info = opcc.get_policy(env_name, pre_trained=1)

    done = False
    env = gym.make(env_name)

    obs = env.reset()
    while not done:
        action = policy(torch.tensor(obs).unsqueeze(0))
        action = action.data.cpu().numpy()[0].astype('float32')
        obs, reward, done, step_info = env.step(action)
        env.render()

