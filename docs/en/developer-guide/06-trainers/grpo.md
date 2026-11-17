English | [简体中文](../../../zh/developer-guide/06-trainers/grpo.md) | [Documentation index](../../index.md)

# GRPO component

`northrl/algorithm/advantage/grpo.py` contains the tensor and complete-trajectory
GRPO formulas, `prepare_grpo_trajectories`, and `GRPOEstimator`. Preparation and
estimation both order trajectories by prompt and run ID, comparing the task ID,
group ID, and `run_id` in that order. Records with equal sort keys retain their
original relative order.
The explicit factory constructs the estimator for `mode: grpo`.
Its `estimate(trajectories, inputs, metrics)` method returns the trajectory list
with scalar advantages written on the original trajectories.

`northrl/algorithm/groups.py` groups repeated rollouts of the same prompt by
`(model_tag, group_id)` and checks for duplicate rollout IDs. When
`expected_rollouts` is supplied, it also checks that every expected rollout slot
appears exactly once. Groups containing failed rollouts are marked to skip;
rewards are checked for finite values in groups that are not skipped. Each
trajectory counts as one complete rollout regardless of how many training
segments it contains.

Training calls `GRPOEstimator.estimate()`, which calls
`compute_grpo_advantage_for_trajectories()`. It accepts `list[Trajectory]`, writes
one scalar advantage to each trajectory in groups that are not skipped, and
returns the original list. Conversion to training data later assigns that
advantage to the tokens in each segment.
The tensor API `compute_grpo_outcome_advantage()` accepts rewards shaped
`[sequence_count, token_count]` and directly returns advantages expanded to token
positions; currently only tests call it. Import both functions directly from
`northrl.algorithm.advantage.grpo`, and import `validate_trajectory_groups` from
`northrl.algorithm.groups`. The obsolete `northrl.algorithm.core_algos` module has
been removed; callers must use these modules directly. The OPD implementation
remains separate. This organization does not change rewards, grouping, sorting,
metrics, or service preparation.

Each successful complete trajectory contributes one reward within its
`(model_tag, group_id)` group, including a trajectory with no training sequence.
The training estimator uses sample standard deviation:
`A = (reward - group_mean) / (group_std + 1e-6)`. A singleton receives zero;
`params.norm_adv_by_std: false` keeps only mean centering. Segments share their
complete trajectory's scalar advantage and do not add observations to the reward statistics.
The rollout checks and skip rules described above run before estimation.

See the [math recipe guide](../../recipe/math/README.md) for existing training
configurations, data requirements, and launch instructions.
