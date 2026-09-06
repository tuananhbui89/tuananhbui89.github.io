def load_balancing_loss(
    gate_logits: tuple[torch.Tensor],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Load balancing loss for Mixture-of-Experts models.


    parameters
    ----------
    layer_logits:
        list with shape (B, T, total_experts) per layer.
    total_experts:
        number of experts inside the moe feed-forward sub-block.
    top_k_experts:
        number of experts chosen per token (k in top-k gating).
    attention_mask:
        optional mask (B, T) where 0 marks padded tokens.

    returns
    -------
    torch.Tensor:
        scalar loss to be added to the training objective.
    """
    # determine device & flat token count
    device = gate_logits[0].device
    num_layers = len(gate_logits)
    bsz, seqlen = attention_mask.shape
    n_tokens = bsz * seqlen

    # merge layers into (tokens, layers, experts)
    stacked = torch.stack(gate_logits, dim=-2).to(device)
    logits = stacked.view(n_tokens, num_layers, num_experts)

    # obtain routing information
    _, routing_probs, sel_idx = route_logits_to_scores(logits, top_k)
    sel_mask = F.one_hot(sel_idx, num_experts)

    if attention_mask is None:
        # average over all tokens
        avg_sel = sel_mask.float().mean(dim=0)
        avg_prob = routing_probs.mean(dim=0)
    else:
        # expand & apply mask
        m_exp = (
            attention_mask.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(bsz, seqlen, num_layers, top_k, num_experts)
            .reshape(-1, num_layers, top_k, num_experts)
        )
        avg_sel = sel_mask.float().mul(m_exp).sum(dim=0) / m_exp.sum(dim=0)

        p_mask = (
            attention_mask.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(bsz, seqlen, num_layers, num_experts)
            .reshape(-1, num_layers, num_experts)
        )
        avg_prob = routing_probs.mul(p_mask).sum(dim=0) / p_mask.sum(dim=0)

    # mismatch penalty
    per_layer = avg_sel * avg_prob.unsqueeze(-2)
    main_loss = per_layer.mean(0).sum() * num_experts

    # --- Minimum usage regularizer: softly penalize underused experts ---
    # avg_sel: (layers, top_k, experts)
    # For each expert, sum over top_k to get total selection per expert per layer
    avg_sel_sum = avg_sel.sum(dim=-2)  # (layers, experts)
    # Normalize so that sum over experts = 1 per layer
    avg_sel_norm = avg_sel_sum / (avg_sel_sum.sum(dim=-1, keepdim=True) + 1e-8)

    # Compute entropy of avg_prob per layer (routing distribution)
    entropy = -(avg_prob * torch.log(avg_prob + 1e-8)).sum(dim=-1)  # (layers,)
    max_entropy = torch.log(torch.tensor(num_experts, dtype=avg_prob.dtype, device=avg_prob.device))
    entropy_scale = 1.5 - entropy / (max_entropy + 1e-8)  # ranges from 0.5 (uniform) to 1.5 (concentrated)

    # Penalty: encourage each expert to be used at least min_threshold
    min_threshold = 0.01 * (64.0 / num_experts)
    
    min_usage_penalty = torch.relu(min_threshold - avg_sel_norm).sum(dim=-1)  # (layers,)
    penalty_coeff = 0.1

    # Final loss: main + entropy-scaled min usage penalty
    return main_loss + penalty_coeff * (min_usage_penalty * entropy_scale).mean()