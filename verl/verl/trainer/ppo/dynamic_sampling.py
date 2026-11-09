from collections import defaultdict

import numpy as np

from verl import DataProto


def filter_reward_groups(batch: DataProto, rollouts_per_prompt: int) -> tuple[DataProto, int, int]:
    """Keep complete prompt groups whose final scalar rewards differ."""
    scores = batch.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Dynamic sampling requires finite rewards.")

    groups = defaultdict(list)
    for index, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups[uid].append(index)

    kept_indices = []
    kept_groups = 0
    for indices in groups.values():
        if len(indices) != rollouts_per_prompt:
            raise ValueError(
                f"Dynamic sampling expected {rollouts_per_prompt} rollouts per prompt, got {len(indices)}."
            )
        if np.ptp(scores[indices]) > 0:
            kept_indices.extend(indices)
            kept_groups += 1

    return batch.select_idxs(np.asarray(kept_indices, dtype=np.int64)), len(groups), kept_groups
