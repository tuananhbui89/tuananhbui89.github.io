# Paper Assets

## Table 1: tab:scope-comparison

Caption: Scope of evolution across application-level self-evolving agentic systems.

| Project | Skill | Prompt | Memory | Harness |
| --- | --- | --- | --- | --- |
| Hermes Agent hermes-github |  |  |  |  |
| SkillClaw ma2026skillclaw |  |  |  |  |
| GenericAgent liang2026genericagent |  |  |  |  |
| EvoAgentX wang2025evoagentx |  |  |  |  |
| MOSS (Ours) |  |  |  |  |

## Figure 1: fig:topology

Caption: MOSS host-side topology. The moss-gateway container hosts the user-facing agent, the in-container evolution service, and a bind-mounted moss CLI; the host-daemon (an asyncio process) serves a Unix-socket RPC for coding-agent and trial-worker orchestration, exposes an HTTP path for evolution control via the gateway, supervises swap requests, and runs the auto-scan engine. The coding-agent CLI is spawned per evolution stage; trial workers are launched per iteration as ephemeral containers. The user-state volume (sessions, memory, credentials, agent configs) is mounted into the moss-gateway container from the host filesystem so state survives an in-place swap (not shown).

No embeddable figure file was copied.

## Figure 2: fig:nesting

Caption: The four nested levels of MOSS evolution: a pre-loop baseline (Layer 0), an iteration loop (Layer 1), a fixed-order seven-stage pipeline per iteration (Layer 2), and internal multi-round loops around the two review gates (Layer 3).

No embeddable figure file was copied.

## Figure 3: fig:iter1-timeline

Caption: Iteration-1 trace covering all stages MOSS executed; stage 3 (Plan-Review) and stage 5 (Code-Review) gates are elided for compactness.

No embeddable figure file was copied.

## Table 2: tab:case-results

Caption: Per-task claweval grader scores (mean of 3 trials per task) before and after the iteration 1 swap.

| Task | Baseline | Iter 1 |  |
| --- | --- | --- | --- |
| T141zh_sla_compliance_audit | 0.3273 | 0.5330 | +0.2057 |
| T142_sla_compliance_audit | 0.2527 | 0.5453 | +0.2926 |
| T137zh_restock_chain_check | 0.2213 | 0.4567 | +0.2354 |
| T138_restock_chain_check | 0.2090 | 0.9049 | +0.6959 |
| mean | 0.2526 | 0.6100 | +0.3574 |
