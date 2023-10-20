import opcc
from collections import defaultdict

env_name = 'd4rl:hammer-v0'
queries = opcc.get_queries(env_name)

policy_info = defaultdict(lambda: defaultdict(lambda: float('-inf')))
for (policy_a_id, policy_b_id), query_batch in queries.items():
    # retrieve policies
    policy_a, _ = opcc.get_policy(*policy_a_id)
    policy_b, _ = opcc.get_policy(*policy_b_id)

    for idx, returns in enumerate(query_batch['info']['return_list_a']):
        policy_info[policy_a_id][query_batch['horizon'][idx]] \
            = max(policy_info[policy_a_id][query_batch['horizon'][idx]],
                  max(returns) - min(returns))

    for idx, returns in enumerate(query_batch['info']['return_list_b']):
        policy_info[policy_b_id][query_batch['horizon'][idx]] \
            = max(policy_info[policy_b_id][query_batch['horizon'][idx]],
                  max(returns) - min(returns))

for horizon in sorted(policy_info[list(policy_info.keys())[0]].keys()):
    for policy in policy_info:
        print(f"Horizon: {horizon},"
              f" Policy: {policy}, "
              f"Max Variance: {policy_info[policy][horizon]}")

    print("===============")
