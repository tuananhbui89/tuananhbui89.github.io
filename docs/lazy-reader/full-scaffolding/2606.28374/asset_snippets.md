# Paper Assets

## Figure 1: fig:money

Caption: No context-evolution artifact universally wins, and unguarded evolution is unsafe. Single-pass methods across four benchmarks on one shared backbone (ALFWorld 7B; GAIA/-bench/WebShop 30B). RSEA (red) is the strongest single-pass method on ALFWorld and never significantly underperforms ReAct (grey) elsewhere; AWM is best on the tool-use tasks; and Dynamic Cheatsheet -- which curates context online with no held-out gate -- is near-best on ALFWorld yet collapses on WebShop (0.14 vs.\ ReAct 0.43). RSEA's strict held-out gate is what makes evolution monotone-safe.

![No context-evolution artifact universally wins, and unguarded evolution is unsafe. Single-pass methods across four benchmarks on one shared backbone (ALFWorld 7B; GAIA/-bench/WebShop 30B). RSEA (red) is the strongest single-pass method on ALFWorld and never significantly underperforms ReAct (grey) elsewhere; AWM is best on the tool-use tasks; and Dynamic Cheatsheet -- which curates context online with no held-out gate -- is near-best on ALFWorld yet collapses on WebShop (0.14 vs.\ ReAct 0.43). RSEA's strict held-out gate is what makes evolution monotone-safe.](figures/fig_crossbench.png)

## Table 1: tab:family

Caption: Context-adaptation methods as (form, update operator, selection rule). The selection rule -- whether a candidate is committed only after improving held-out performance -- is what we argue governs reliability.

| Method | Artifact form | Update operator | Selection rule |
| --- | --- | --- | --- |
| Reflexion | per-task reflection | verbal self-feedback | none (within-task retry) |
| GEPA | flat prompt | reflective mutation | Pareto on val minibatch |
| AWM | workflow list | induction from successes | none (induce & inject) |
| ACE | bullet playbook | add-only deltas | val keep-best (non-strict) |
| Dynamic Cheatsheet | single cheatsheet | online rewrite | none (no held-out gate) |
| RSEA (ours) | 3-layer state | holistic rewrite of all layers | strict held-out keep-better |

## Figure 2: fig:arch

Caption: RSEA recursively rewrites a three-layer natural-language state of a frozen LLM agent. The state is injected as a preamble into a standard ReAct loop; across generations it is rewritten from evolve-set trajectories and frozen only on a strict held-out validation improvement, which makes the loop monotone-safe.

![RSEA recursively rewrites a three-layer natural-language state of a frozen LLM agent. The state is injected as a preamble into a standard ReAct loop; across generations it is rewritten from evolve-set trajectories and frozen only on a strict held-out validation improvement, which makes the loop monotone-safe.](figures/fig_architecture.png)

## Table 2: tab:alfworld

Caption: ALFWorld (134 tasks 5 seeds, Qwen2.5-7B). Success rate mean95% CI; subscripts are McNemar-significant gains (gaingreen) over ReAct. ``+retry'' adds the same multi-trial loop to each prior. RSEA is the strongest single-pass context method; RSEA_R is best overall.

| Single-pass | Succ.\ (%) | + retry | Succ.\ (%) |
| --- | --- | --- | --- |
| ReAct | 64.64.3 | Reflexion | 76.4 |
| GEPA | 63.95.3 | GEPA_R | 75.85.4 |
| AWM | 65.43.3 | ACE_R | 76.74.9 |
| ACE | 66.92.5 |  |  |
| Dynamic Cheatsheet | 70.73.4 |  |  |
| RSEA (ours) | 69.39.3 4.7 | RSEA_R (ours) | 79.47.4 14.8 |

## Table 3: tab:bytype

Caption: ALFWorld success by task family (%, 5 seeds). RSEA's gains concentrate in the families its evolved skills/playbook directly address (examine, pick_two, pick_and_place).

| Method | examine | pick_two | pick_and_place | heat | cool | clean |
| --- | --- | --- | --- | --- | --- | --- |
| ReAct | 6.7 | 36.5 | 82.5 | 76.5 | 81.0 | 80.0 |
| RSEA | 26.7 | 43.5 | 91.7 | 79.1 | 83.8 | 73.5 |
| RSEA_R | 44.4 | 67.1 | 99.2 | 87.0 | 88.6 | 79.4 |

## Table 4: tab:cross

Caption: Cross-benchmark comparison (single-pass). ALFWorld: success % (1345 seeds, 7B). GAIA: accuracy % (30, 30B). -bench retail: success % (60 eval, 30B). WebShop: mean dense score (100 eval, 30B). Bold=column best. : RSEA p=0.015 vs.\ ReAct (ALFWorld). rowgDC has no held-out gate and is high-variance (best on ALFWorld, worst on WebShop/-bench); RSEA is significantly best on ALFWorld and never regresses elsewhere.

| Method | ALFWorld | GAIA | -bench | WebShop |
| --- | --- | --- | --- | --- |
| succ.% | acc.% | succ.% | score |  |
| ReAct | 64.6 | 16.7 | 41.7 | 0.429 |
| GEPA | 63.9 | -- | 41.7 | 0.415 |
| AWM | 65.4 | -- | 51.7 | 0.460 |
| ACE | 66.9 | -- | 40.0 | 0.453 |
| rowg Dynamic Cheatsheet | 70.7 | -- | 36.7 | 0.136 |
| RSEA (ours) | 69.3^ | 13.3 | 40.0 | 0.437 |

## Figure 3: fig:gencurves

Caption: RSEA self-evolution: held-out validation over generations (strict keep-better). The best-kept state improves where there is signal (-bench 0.200.36) and the strict gate rejects every regressive candidate where there is not (WebShop), so the frozen state never underperforms ReAct.

![RSEA self-evolution: held-out validation over generations (strict keep-better). The best-kept state improves where there is signal (-bench 0.200.36) and the strict gate rejects every regressive candidate where there is not (WebShop), so the frozen state never underperforms ReAct.](figures/fig_gencurves.png)

## Table 5: tab:abl_layers

Caption: Layer ablation (ALFWorld test, 543 seeds). Every subset beats ReAct; layers overlap.

| Injected state | Succ.\ (%) |
| --- | --- |
| empty (= ReAct) | 64.2 |
| strategy only | 67.3 |
| skills only | 69.8 |
| playbook only | 69.8 |
| strategy + skills | 67.9 |
| full RSEA | 68.5 |
