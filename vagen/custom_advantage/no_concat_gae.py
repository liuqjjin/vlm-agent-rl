import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est


def _to_numpy_int64(x, factorize_if_non_numeric: bool = False):
    """Convert input to np.ndarray(dtype=int64).

    Args:
        x: Input data (torch.Tensor, numpy array, or array-like).
        factorize_if_non_numeric: If True, encode non-numeric values (strings, UUIDs)
            into contiguous int64 codes using stable factorization (default: False).

    Returns:
        np.ndarray: Integer array with dtype int64.

    Raises:
        ValueError: If x contains non-numeric data and factorize_if_non_numeric is False.

    Example:
        >>> _to_numpy_int64([1, 2, 3])
        array([1, 2, 3], dtype=int64)
        >>> _to_numpy_int64(["uuid-a", "uuid-b", "uuid-a"], factorize_if_non_numeric=True)
        array([0, 1, 0], dtype=int64)

    Note:
        Factorization assigns the same code to identical values within a batch,
        enabling use of string-based group identifiers in GAE computation.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # Fast path: already integer
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64, copy=False)

    # If object/strings: try numeric cast first, otherwise factorize
    if x.dtype == np.object_ or np.issubdtype(x.dtype, np.str_):
        try:
            return x.astype(np.int64)
        except (ValueError, TypeError):
            if not factorize_if_non_numeric:
                # Give a clearer error
                sample = x.reshape(-1)[:5]
                raise ValueError(
                    f"Cannot cast to int64. Sample values: {sample}. "
                    f"Set factorize_if_non_numeric=True for string-like IDs."
                )
            # Factorize: same original value -> same code
            _, inv = np.unique(x, return_inverse=True)
            return inv.astype(np.int64, copy=False)

    # Fallback: try casting for float/bool/etc.
    return x.astype(np.int64, copy=False)


@register_adv_est("no_concat_gae_last")
def compute_gae_no_concat_advantage_return(
    data,
    gamma,
    lam,
    config=None,
    ignore_value: float = -100.0,
    **kwargs,
):
    """Compute token-level advantages and returns for no-concat GAE.

    Implements Generalized Advantage Estimation (GAE) for multi-turn trajectories
    with distributed padding support. Value baseline is taken from the last valid
    response token per turn, and returns are supervised only at that position.

    Args:
        data: Rollout data object containing batch tensors and non-tensor metadata.
            Required fields:
            - batch["token_level_scores"]: (batch_size, seq_len) per-token rewards
            - batch["response_mask"]: (batch_size, seq_len) binary mask for response tokens
            - batch.get("values"): (batch_size, seq_len) value predictions (default: zeros)
            - non_tensor_batch["group_idx"]: group identifiers (supports UUIDs)
            - non_tensor_batch["traj_idx"]: trajectory index within group
            - non_tensor_batch["turn_idx"]: turn index within trajectory
        gamma: Discount factor for future rewards (typically 0.99).
        lam: GAE lambda parameter for bias-variance tradeoff (typically 0.95).
        config: Optional configuration (unused).
        ignore_value: Fill value for non-supervised return positions (default: -100.0).
        **kwargs: Additional arguments (unused).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (advantages, returns) both of shape (batch_size, seq_len).
            - advantages: whitened GAE advantages broadcast across all valid response tokens
            - returns: supervised returns at last valid token per turn, ignore_value elsewhere

    Example:
        >>> # data with 2 turns, each turn has 3 response tokens
        >>> advantages, returns = compute_gae_no_concat_advantage_return(
        ...     data, gamma=0.99, lam=0.95
        ... )
        >>> # returns[i, j] == ignore_value except at last valid token positions
        >>> # advantages are whitened and spread across all response tokens per turn

    References:
        Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation". This implementation handles multi-turn episodes with
        automatic deduplication of distributed padding artifacts.

    Note:
        - Padding duplicates (identical group/traj/turn) are automatically detected and
          deduplicated before GAE computation, then broadcast back to full batch.
        - Turn-level value is extracted from the LAST valid response token per turn.
        - GAE is computed backward through turns within each (group, trajectory) pair.
    """
    token_level_scores = data.batch["token_level_scores"]
    values = data.batch.get("values", torch.zeros_like(token_level_scores))
    response_mask = data.batch["response_mask"]
    group_idx = data.non_tensor_batch["group_idx"]
    turn_idx = data.non_tensor_batch["turn_idx"]
    traj_idx = data.non_tensor_batch["traj_idx"]

    with torch.no_grad():
        device = token_level_scores.device
        bs, L = token_level_scores.shape

        # Floating mask for arithmetic, boolean mask for indexing
        mask_f_full = response_mask.to(dtype=token_level_scores.dtype, device=device)
        mask_b_full = response_mask.to(dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # 0) Handle padded duplicates:
        #    identical (group_idx, traj_idx, turn_idx) are padding artifacts.
        #    Compute once per unique triple, then broadcast back.
        #
        #    IMPORTANT: group_idx may be UUID/string, so we factorize it.
        # ------------------------------------------------------------------
        g_np = _to_numpy_int64(group_idx, factorize_if_non_numeric=True)
        t_np = _to_numpy_int64(traj_idx, factorize_if_non_numeric=False)
        u_np = _to_numpy_int64(turn_idx, factorize_if_non_numeric=False)

        key3 = np.stack([g_np, t_np, u_np], axis=1)  # (bs, 3) int64

        uniq_key3, uniq_first_idx, inverse = np.unique(
            key3, axis=0, return_index=True, return_inverse=True
        )
        uniq_first_idx = uniq_first_idx.astype(np.int64, copy=False)
        bs_u = uniq_first_idx.shape[0]

        uniq_first_idx_t = torch.as_tensor(uniq_first_idx, dtype=torch.long, device=device)

        scores = token_level_scores.index_select(0, uniq_first_idx_t)  # (bs_u, L)
        vals   = values.index_select(0, uniq_first_idx_t)              # (bs_u, L)
        mask_f = mask_f_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)
        mask_b = mask_b_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)

        group_ids = torch.as_tensor(uniq_key3[:, 0], dtype=torch.long, device="cpu")
        traj_ids  = torch.as_tensor(uniq_key3[:, 1], dtype=torch.long, device="cpu")
        turn_ids  = torch.as_tensor(uniq_key3[:, 2], dtype=torch.long, device="cpu")

        # ------------------------------------------------------------------
        # 1) Turn-level reward: sum token-level scores within the response
        # ------------------------------------------------------------------
        turn_rewards = (scores * mask_f).sum(dim=1)  # (bs_u,)

        # ------------------------------------------------------------------
        # 2) Turn-level value: value at the last valid token (vectorized)
        # ------------------------------------------------------------------
        # last_pos[b] = last index where mask_b[b] is True; -1 if the row has no valid tokens
        valid_counts = mask_b.sum(dim=1)  # (bs_u,)
        arange_L = torch.arange(L, device=device).view(1, L).expand(bs_u, L)

        last_pos = (mask_b.to(torch.long) * arange_L).max(dim=1).values
        last_pos = torch.where(valid_counts > 0, last_pos, torch.full_like(last_pos, -1))

        # turn_values[b] = vals[b, last_pos[b]] if last_pos[b] >= 0 else 0
        gather_pos = last_pos.clamp(min=0).view(bs_u, 1)
        gathered = vals.gather(1, gather_pos).squeeze(1)

        turn_values = torch.zeros(bs_u, dtype=vals.dtype, device=device)
        turn_values = torch.where(last_pos >= 0, gathered, turn_values)

        # ------------------------------------------------------------------
        # 3) Compute turn-level GAE grouped by (group_idx, traj_idx)
        # ------------------------------------------------------------------
        # Advantages: kept token-level (filled on valid tokens as before).
        advantages_u = torch.zeros((bs_u, L), dtype=vals.dtype, device=device)

        # Returns: initialize everything as ignore_value, and only write return
        # at the *supervised* position(s) (here: last valid token per response).
        returns_u = torch.full((bs_u, L), float(ignore_value), dtype=vals.dtype, device=device)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        # group by (group_id, traj_id); within each group, order by turn_id
        key2 = torch.stack([group_ids, traj_ids], dim=1)
        unique_key2 = torch.unique(key2, dim=0)

        for g_id, tr_id in unique_key2.tolist():
            pair_mask = torch.nonzero(
                (group_ids == g_id) & (traj_ids == tr_id),
                as_tuple=False
            ).view(-1)

            if pair_mask.numel() == 0:
                continue

            sorted_turns = pair_mask[torch.argsort(turn_ids[pair_mask])].tolist()

            next_value = 0.0
            lastgaelam = 0.0

            # Backward recursion over turns in the trajectory
            for b in reversed(sorted_turns):
                r = float(turn_rewards[b].item())
                v = float(turn_values[b].item())

                delta = r + gamma_val * next_value - v
                lastgaelam = delta + gamma_val * lam_val * lastgaelam

                adv = lastgaelam
                ret = adv + v

                lp = int(last_pos[b].item())
                if lp >= 0:
                    # Advantage is applied across all valid response tokens (existing behavior)
                    advantages_u[b, mask_b[b]] = adv

                    # Return supervision only at last valid token;
                    # all other tokens remain ignore_value.
                    returns_u[b, lp] = ret

                next_value = v

        # ------------------------------------------------------------------
        # 4) Optional whitening (unique samples only)
        # ------------------------------------------------------------------
        advantages_u = verl_F.masked_whiten(advantages_u, mask_f)

        # ------------------------------------------------------------------
        # 5) Broadcast back to full batch (including padded duplicates)
        # ------------------------------------------------------------------
        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages_full = advantages_u.index_select(0, inverse_t)  # (bs, L)
        returns_full = returns_u.index_select(0, inverse_t)        # (bs, L)

    return advantages_full, returns_full


@register_adv_est("no_concat_gae")
def compute_gae_no_concat_advantage_return_firsttok(
    data,
    gamma,
    lam,
    config=None,
    ignore_value: float = -100.0,
    **kwargs,
):
    """no-concat GAE variant that uses the FIRST valid response token as the per-turn value anchor.

    Differences vs "no_concat_gae_last":
    - Turn value v_t is taken from the FIRST valid response token (not the last).
    - The computed return for the turn is written to the FIRST valid response token position.
    - Advantages are still broadcast across all valid response tokens in the turn (same as before).
    - All non-supervised return positions remain `ignore_value` so downstream value_mask can be
      derived from returns.

    Args:
        data: Rollout data object containing batch tensors and non-tensor metadata.
            Required fields:
            - batch["token_level_scores"]: (batch_size, seq_len) per-token rewards
            - batch["response_mask"]: (batch_size, seq_len) binary mask for response tokens
            - batch.get("values"): (batch_size, seq_len) value predictions (default: zeros)
            - non_tensor_batch["group_idx"]: group identifiers (supports UUIDs)
            - non_tensor_batch["traj_idx"]: trajectory index within group
            - non_tensor_batch["turn_idx"]: turn index within trajectory
        gamma: Discount factor for future rewards (typically 0.99).
        lam: GAE lambda parameter for bias-variance tradeoff (typically 0.95).
        config: Optional configuration (unused).
        ignore_value: Fill value for non-supervised return positions (default: -100.0).
        **kwargs: Additional arguments (unused).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (advantages, returns) both of shape (batch_size, seq_len).
            - advantages: whitened GAE advantages broadcast across all valid response tokens
            - returns: supervised returns at first valid token per turn, ignore_value elsewhere

    Example:
        >>> # Prefer this variant when value head predicts from first token embeddings
        >>> advantages, returns = compute_gae_no_concat_advantage_return_firsttok(
        ...     data, gamma=0.99, lam=0.95
        ... )
        >>> # returns[i, first_valid_pos] contains supervised target
        >>> # returns[i, other_pos] == ignore_value

    Shape contracts:
        - token_level_scores, values, response_mask: (bs, L)
        - returns_full: (bs, L) filled with ignore_value except first valid token positions

    References:
        Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation". This variant aligns value supervision with first-token
        value prediction architectures.

    Note:
        Use this when your value head processes the first response token embedding.
        Use "no_concat_gae_last" when the value head processes the last token embedding.
    """
    token_level_scores = data.batch["token_level_scores"]
    values = data.batch.get("values", torch.zeros_like(token_level_scores))
    response_mask = data.batch["response_mask"]
    group_idx = data.non_tensor_batch["group_idx"]
    turn_idx = data.non_tensor_batch["turn_idx"]
    traj_idx = data.non_tensor_batch["traj_idx"]

    with torch.no_grad():
        device = token_level_scores.device
        bs, L = token_level_scores.shape

        mask_f_full = response_mask.to(dtype=token_level_scores.dtype, device=device)
        mask_b_full = response_mask.to(dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # 0) Handle padded duplicates (unique by (group, traj, turn))
        # ------------------------------------------------------------------
        g_np = _to_numpy_int64(group_idx, factorize_if_non_numeric=True)
        t_np = _to_numpy_int64(traj_idx, factorize_if_non_numeric=False)
        u_np = _to_numpy_int64(turn_idx, factorize_if_non_numeric=False)

        key3 = np.stack([g_np, t_np, u_np], axis=1)  # (bs, 3)

        uniq_key3, uniq_first_idx, inverse = np.unique(
            key3, axis=0, return_index=True, return_inverse=True
        )
        uniq_first_idx = uniq_first_idx.astype(np.int64, copy=False)
        bs_u = uniq_first_idx.shape[0]

        uniq_first_idx_t = torch.as_tensor(uniq_first_idx, dtype=torch.long, device=device)

        scores = token_level_scores.index_select(0, uniq_first_idx_t)  # (bs_u, L)
        vals   = values.index_select(0, uniq_first_idx_t)              # (bs_u, L)
        mask_f = mask_f_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)
        mask_b = mask_b_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)

        group_ids = torch.as_tensor(uniq_key3[:, 0], dtype=torch.long, device="cpu")
        traj_ids  = torch.as_tensor(uniq_key3[:, 1], dtype=torch.long, device="cpu")
        turn_ids  = torch.as_tensor(uniq_key3[:, 2], dtype=torch.long, device="cpu")

        # ------------------------------------------------------------------
        # 1) Turn-level reward: sum of token-level scores within the response
        # ------------------------------------------------------------------
        turn_rewards = (scores * mask_f).sum(dim=1)  # (bs_u,)

        # ------------------------------------------------------------------
        # 2) Turn-level value: value at the FIRST valid token (vectorized)
        # ------------------------------------------------------------------
        # first_pos[b] = first index where mask_b[b] is True; -1 if no valid tokens
        has_any = mask_b.any(dim=1)  # (bs_u,)
        arange_L = torch.arange(L, device=device).view(1, L).expand(bs_u, L)

        # Use a large sentinel for invalid positions, then take min
        big = torch.full_like(arange_L, L)
        first_pos = torch.where(mask_b, arange_L, big).min(dim=1).values  # in [0..L] possibly L
        first_pos = torch.where(has_any, first_pos, torch.full_like(first_pos, -1))

        # turn_values[b] = vals[b, first_pos[b]] if first_pos[b] >= 0 else 0
        gather_pos = first_pos.clamp(min=0).view(bs_u, 1)
        gathered = vals.gather(1, gather_pos).squeeze(1)

        turn_values = torch.zeros(bs_u, dtype=vals.dtype, device=device)
        turn_values = torch.where(first_pos >= 0, gathered, turn_values)

        # ------------------------------------------------------------------
        # 3) Compute turn-level GAE grouped by (group_idx, traj_idx)
        # ------------------------------------------------------------------
        advantages_u = torch.zeros((bs_u, L), dtype=vals.dtype, device=device)
        returns_u = torch.full((bs_u, L), float(ignore_value), dtype=vals.dtype, device=device)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        key2 = torch.stack([group_ids, traj_ids], dim=1)
        unique_key2 = torch.unique(key2, dim=0)

        for g_id, tr_id in unique_key2.tolist():
            pair_mask = torch.nonzero(
                (group_ids == g_id) & (traj_ids == tr_id),
                as_tuple=False
            ).view(-1)

            if pair_mask.numel() == 0:
                continue

            sorted_turns = pair_mask[torch.argsort(turn_ids[pair_mask])].tolist()

            next_value = 0.0
            lastgaelam = 0.0

            for b in reversed(sorted_turns):
                r = float(turn_rewards[b].item())
                v = float(turn_values[b].item())

                delta = r + gamma_val * next_value - v
                lastgaelam = delta + gamma_val * lam_val * lastgaelam

                adv = lastgaelam
                ret = adv + v

                fp = int(first_pos[b].item())
                if fp >= 0:
                    # same behavior: broadcast advantage across all valid response tokens
                    advantages_u[b, mask_b[b]] = adv

                    # return supervision at FIRST valid token
                    returns_u[b, fp] = ret

                next_value = v

        # ------------------------------------------------------------------
        # 4) Optional whitening (unique samples only)
        # ------------------------------------------------------------------
        advantages_u = verl_F.masked_whiten(advantages_u, mask_f)

        # ------------------------------------------------------------------
        # 5) Broadcast back to full batch (including padded duplicates)
        # ------------------------------------------------------------------
        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages_full = advantages_u.index_select(0, inverse_t)  # (bs, L)
        returns_full = returns_u.index_select(0, inverse_t)        # (bs, L)

    return advantages_full, returns_full
