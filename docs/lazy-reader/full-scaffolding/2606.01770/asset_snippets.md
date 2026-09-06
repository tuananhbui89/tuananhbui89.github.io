# Paper Assets

## Figure 1: fig:c1

Caption: Longer frequent evolution can overfit earlier stream evidence. Top: The A-Evolve run with unbounded evolution grows from 12 to 34 skills, while the prompt grows from 2 KB to 68 KB. Bottom: Each curve reports pass-rate increase relative to the no-evolution solver, with different evolution stopping cycles. Early gains fade as later tasks arrive; news_from_future.md helps on a sports task yet misfires on a politics task.

![Longer frequent evolution can overfit earlier stream evidence. Top: The A-Evolve run with unbounded evolution grows from 12 to 34 skills, while the prompt grows from 2 KB to 68 KB. Bottom: Each curve reports pass-rate increase relative to the no-evolution solver, with different evolution stopping cycles. Early gains fade as later tasks arrive; news_from_future.md helps on a sports task yet misfires on a politics task.](figures/figure_c1_paper.png)

## Figure 2: fig:teaser

Caption: Three deployment dimensions in open-ended task streams. Unbounded stream, heterogeneous tasks, and non-stationary distributions expose the limits of evolving a single dense harness for long-term deployment

![Three deployment dimensions in open-ended task streams. Unbounded stream, heterogeneous tasks, and non-stationary distributions expose the limits of evolving a single dense harness for long-term deployment](figures/teaser.png)

## Figure 3: fig:unified_pipeline

Caption: Overview of the Adaptive Auto-Harness system. Top: the multi-agent evolver constructs and refines a harness tree across cycles via four phases (Analyze Research Build Verify) with a persistent cross-cycle workspace and temporal-reveal feedback. Bottom: at solve time, a router agent reads each branch's workspace via git show and routes the incoming task x_t to the most suitable branch. Two human-in-the-loop hooks (task-board steering and research-phase assistance) trigger only when the evolver's history lacks relevant signal.

![Overview of the Adaptive Auto-Harness system. Top: the multi-agent evolver constructs and refines a harness tree across cycles via four phases (Analyze Research Build Verify) with a persistent cross-cycle workspace and temporal-reveal feedback. Bottom: at solve time, a router agent reads each branch's workspace via git show and routes the incoming task x_t to the most suitable branch. Two human-in-the-loop hooks (task-board steering and research-phase assistance) trigger only when the evolver's history lacks relevant signal.](figures/framework.png)

## Table 1: tab:bench_stats

Caption: Benchmark statistics for the three chronological task streams.

| lrlll@ Bench. | Tasks | Span | Domain |
| --- | --- | --- | --- |
| PolyBench | 5,075 | Feb 6--22, 2026 | Prediction markets |
| CTF-Dojo | 261 | 2011--2024 | Security |
| FutureX | 503 | Jan--Apr 2026 | Forecasting |

## Table 2: tab:rq1_main

Caption: Main comparison across no-evolution agents, auto-harness baselines, a human-designed system, and our three variants. PolyBench reports Accuracy / Return (coverage-scaled CWR, in %); CTF-Dojo and FutureX report the official Pass@1. Bold marks the best result per row and underline marks second best.

| llccccccccccccccc@ 2c | 5cNo evolution | 5cAuto-harness baselines | 1cHuman | 3cOurs |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (lr)3-7 (lr)8-12 (lr)13-13 (lr)14-16 Benchmark | Metric | Sonnet | Haiku | DeepSeek | Kimi | GLM | A-Evolve | GEPA |  |  |  |  |  |  |  |
| Harness | . |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Harness | SkillOS | OctoTools |  |  |  |  |  |  |  |  |  |  |  |  |  |
| agent | Adaptive |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| System |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2*PolyBench | Accuracy | 22.2 | 15.2 | 1.4 | 14.0 | 14.0 | 18.4 | 13.4 | 50.8 | 8.5 | 21.4 | 40.0 | 79.8 | 77.4 | 80.9 |
| Return | +1.7 | +88.0 | +16.4 | -2.0 | +10.1 | +7.2 | +0.2 | +320 | +1.7 | +3.6 | +20.4 | +351 | +352 | +330 |  |
| CTF-Dojo | Pass | 37.2 | 23.8 | 26.1 | 24.5 | 12.6 | 45.2 | 42.9 | 41.0 | 25.7 | 29.5 | 38.3 | 47.9 | 46.0 | 50.2 |
| FutureX | Pass | 31.0 | 31.0 | 31.2 | 27.8 | 30.8 | 47.5 | 28.2 | 29.4 | 31.8 | 29.8 | 25.6 | 49.5 | 44.1 | 47.3 |

## Figure 4: fig:l_evo_evidence

Caption: evidence across benchmark-specific bottlenecks. PolyBench stresses confidence calibration, FutureX stresses web-retrieval access, and CTF-Dojo stresses payload handling of different file sizes.

![evidence across benchmark-specific bottlenecks. PolyBench stresses confidence calibration, FutureX stresses web-retrieval access, and CTF-Dojo stresses payload handling of different file sizes.](figures/l_evo_evidence.png)

## Figure 5: fig:l_adapt_evidence

Caption: evidence across task categories. Curves show cumulative adaptation lift over the baselines, while shaded bands show performance spread across task categories over cycles.

![evidence across task categories. Curves show cumulative adaptation lift over the baselines, while shaded bands show performance spread across task categories over cycles.](figures/l_adapt_evidence.png)

## Figure 6: fig:rq2_stats

Caption: Ablations for multi-agent evolution. Removing evaluation feedback or cross-cycle memory degrades performance relative to the full stateful system.

![Ablations for multi-agent evolution. Removing evaluation feedback or cross-cycle memory degrades performance relative to the full stateful system.](figures/figure_rq2.png)

## Figure 7: fig:rq3_stats

Caption: Designed analysis of solve-time routing on an evolved harness tree. We seed one branch per task category, evolve the tree over the stream, and replay every task through every branch. Oracle is the best branch per task, Adapt a category-based routing policy, Naive the main workspace only, and Worst the worst branch per task.

![Designed analysis of solve-time routing on an evolved harness tree. We seed one branch per task category, evolve the tree over the stream, and replay every task through every branch. Oracle is the best branch per task, Adapt a category-based routing policy, Naive the main workspace only, and Worst the worst branch per task.](figures/figure_rq3.png)

## Figure 8: fig:mechanism_traces

Caption: Extracted trajectories of the designed multi-agent evolution, agentic routing, and Human-in-the-Loop.

![Extracted trajectories of the designed multi-agent evolution, agentic routing, and Human-in-the-Loop.](figures/rq2_demo.png)

## Figure 9: fig:rq4_hitl_passrate

Caption: FutureX pass-rate lift over four task slices under two human-steering hooks. The orange triangle marks research-phase steering and the purple triangle marks task-board steering. Lift is measured against the no-HITL run.

![FutureX pass-rate lift over four task slices under two human-steering hooks. The orange triangle marks research-phase steering and the purple triangle marks task-board steering. Lift is measured against the no-HITL run.](figures/figure_rq4_timeline.png)

## Table 3: tab:hyperparams

Caption: Adaptive Auto-Harness hyperparameters across the three benchmarks. EGL is the Expected-Gain-from-Learning trigger that gates whether a cycle runs. All runs use T=0 for both solver and evolver to attribute gains to the algorithm rather than sampling noise.

| lccc@ Hyperparameter | PolyBench | CTF-Dojo | FutureX |
| --- | --- | --- | --- |
| 4@lModels |  |  |  |
| Solver | Sonnet 4.6 | Sonnet 4.6 | Sonnet 4.6 |
| Evolver | Opus 4.6 | Opus 4.6 | Opus 4.6 |
| Router | Sonnet 4.6 | Sonnet 4.6 | Sonnet 4.6 |
| 4@lSampling & budget |  |  |  |
| Solver temperature | 0.0 | 0.0 | 0.0 |
| Evolver temperature | 0.0 | 0.0 | 0.0 |
| Solver max turns | 80 | 80 | 80 |
| Evolver max tokens | 128k | 128k | 128k |
| 4@lStream & schedule |  |  |  |
| Total tasks | 5,075 | 261 | 503 |
| Batch size | 100 | 20 | 20 |
| Evolution cycles | 51 | 14 | 26 |
| EGL threshold | 0.05 | 0.05 | 0.05 |
| EGL window | 3 | 3 | 3 |
| Solve workers | 24 | 8 | 10 |
| 4@lMulti-agent & routing |  |  |  |
| Research parallel agents | 3 | 3 | 3 |
| Build/verify retries | 3 | 3 | 3 |
| Routing confidence threshold | 0.7 | 0.7 | 0.7 |
| 4@lSandbox |  |  |  |
| Solver sandbox network | none | none | bridge |
| Evolver sandbox network | none | none | bridge |

## Table 4: tab:seed_inventory

Caption: Seed harness shipped to the evolver before any cycle runs. Skills counts top-level skill directories; Memory counts JSONL entries; Infra indicates whether the seed includes an infrastructure directory.

| lrrrrl@ Benchmark | Prompt LOC | Skills | Tools | Memory | Infra |
| --- | --- | --- | --- | --- | --- |
| PolyBench | 27 | 0 | 0 | 0 | no |
| CTF-Dojo | 24 | 0 | 0 | 0 | no |
| FutureX | 114 | 0 | 0 | 0 | yes |

## Table 5: tab:cost

Caption: Solver token cost and wall-clock per system. Tokens are summed from per-task input_tokens / output_tokens in results.jsonl. Wall-clock sums per-task elapsed seconds and excludes orchestration overhead. Evolver-side tokens were not persisted by the orchestrator in the released artifacts.

| llrrrr@ System | Bench | In (M tok) | Out (M tok) | Hours | Tasks/h |
| --- | --- | --- | --- | --- | --- |
| Sonnet (no-evo) | PolyBench | 35.3 | 5.0 | 25.6 | 198.3 |
| CTF-Dojo | 138.4 | 2.1 | 22.4 | 11.7 |  |
| FutureX | 55.5 | 0.7 | 34.2 | 14.7 |  |
| A-Evolve | PolyBench | 365.4 | 4.2 | 23.9 | 212.3 |
| CTF-Dojo | 111.3 | 1.7 | 11.4 | 23.0 |  |
| FutureX | 268.8 | 1.2 | 13.8 | 36.5 |  |
| Meta-Harness | PolyBench | 245.0 | 9.6 | 49.7 | 102.1 |
| CTF-Dojo | 142.7 | 2.2 | 20.0 | 13.0 |  |
| FutureX | 39.1 | 0.5 | 7.9 | 63.8 |  |
| Multi-agent | PolyBench | 264.2 | 14.5 | 78.0 | 65.1 |
| CTF-Dojo | 180.2 | 2.5 | 21.2 | 12.3 |  |
| FutureX | 130.7 | 1.0 | 12.1 | 41.7 |  |
| Full System | PolyBench | 233.2 | 12.2 | 59.5 | 85.2 |
| CTF-Dojo | 169.0 | 2.4 | 21.1 | 12.4 |  |
| FutureX | 25.6 | 0.5 | 6.6 | 75.7 |  |

## Figure 10: fig:app_c2_evolver_budget

Caption: Evolver capability on CTF-Dojo. Pass rate improves with stronger evolver models; higher budget helps Haiku and Sonnet but gives little additional gain once Opus already reaches high performance.

![Evolver capability on CTF-Dojo. Pass rate improves with stronger evolver models; higher budget helps Haiku and Sonnet but gives little additional gain once Opus already reaches high performance.](figures/figure_c2_final.png)

## Figure 11: fig:app_polybench_dilution

Caption: PolyBench workspace dilution. A PolyBench-evolved workspace reaches the highest CWR, while combining all evolved workspaces sharply reduces CWR, supporting the need for specialized harness branches rather than a single dense harness.

![PolyBench workspace dilution. A PolyBench-evolved workspace reaches the highest CWR, while combining all evolved workspaces sharply reduces CWR, supporting the need for specialized harness branches rather than a single dense harness.](figures/figure_c3_polybench_dilution.png)

## Figure 12: fig:app_nonstationarity_polybench

Caption: PolyBench non-stationarity. Market difficulty and tradability shift over time: later markets are less often decisive or liquid and more often near-even.

![PolyBench non-stationarity. Market difficulty and tradability shift over time: later markets are less often decisive or liquid and more often near-even.](figures/fig1_strategy_drift.png)

## Figure 13: fig:app_nonstationarity_ctf_dojo

Caption: CTF-Dojo non-stationarity. The chronological stream keeps introducing new competitions and increases cross-competition variability, so early challenge experience is not uniformly transferable.

![CTF-Dojo non-stationarity. The chronological stream keeps introducing new competitions and increases cross-competition variability, so early challenge experience is not uniformly transferable.](figures/shift_crypto.png)

## Figure 14: fig:app_nonstationarity_futurex

Caption: FutureX non-stationarity. Language, source accessibility, difficulty, and answer format shift across batches, creating solve-time harness mismatch when one static harness is reused for all tasks.

![FutureX non-stationarity. Language, source accessibility, difficulty, and answer format shift across batches, creating solve-time harness mismatch when one static harness is reused for all tasks.](figures/property_shift.png)

## Table 6: tab:per_domain_ctf

Caption: Per-category Pass@1 (%) on CTF-Dojo. Categories are parsed from the detail field; binary/pwn merges the conventionally-equivalent CTF tags. Bold marks the best system per row.

| lrrrrrrr@ Category | N | Sonnet | A-Evolve | Meta-H. | Multi | Adapt. | Full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crypto | 74 | 52.7 | 66.1 | 56.2 | 67.6 | 55.6 | 72.1 |
| binary/pwn | 41 | 4.9 | 11.5 | 2.5 | 7.0 | 4.8 | 14.8 |
| web | 11 | 45.5 | 41.7 | 41.7 | 33.3 | 50.0 | 72.7 |
| reverse | 65 | 49.2 | 62.7 | 48.7 | 59.7 | 57.0 | 66.2 |
| forensics | 12 | 58.3 | 61.5 | 64.3 | 64.3 | 71.4 | 58.3 |
| misc | 58 | 20.7 | 23.5 | 29.5 | 34.1 | 40.5 | 25.6 |
| Overall | 261 | 37.2 | 45.2 | 41.0 | 47.9 | 46.0 | 50.2 |

## Table 7: tab:per_domain_futurex

Caption: Per-slice Pass@1 (%) on FutureX. Language is detected from Chinese characters in the question; domain is inferred by keyword match (other catches unmatched questions). N is from the Sonnet baseline run. The zh, geopolitics row contains a single task and is reported for completeness only. Bold marks the best system per row.

| llrrrrrrr@ Lang | Domain | N | Sonnet | A-Evolve | Meta-H. | Multi | Adapt. | Full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | finance | 76 | 21.1 | 53.9 | 21.1 | 52.6 | 43.4 | 56.6 |
| en | tech | 8 | 25.0 | 50.0 | 37.5 | 50.0 | 37.5 | 50.0 |
| en | geopolitics | 43 | 41.9 | 62.8 | 37.2 | 62.8 | 60.5 | 65.1 |
| en | sports | 62 | 38.7 | 58.1 | 37.1 | 59.7 | 56.5 | 54.8 |
| en | entertainment | 37 | 27.0 | 43.2 | 21.6 | 40.5 | 48.6 | 37.8 |
| en | other | 219 | 38.8 | 52.1 | 37.4 | 55.7 | 48.9 | 51.1 |
| zh | finance | 10 | 0.0 | 0.0 | 0.0 | 30.0 | 0.0 | 20.0 |
| zh | geopolitics | 1 | 100 | 100 | 0 | 0 | 0 | 0 |
| zh | entertainment | 25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| zh | other | 22 | 0.0 | 0.0 | 0.0 | 4.5 | 0.0 | 4.5 |
| Overall | 503 | 31.0 | 47.5 | 29.4 | 49.5 | 44.1 | 47.3 |  |

## Table 8: tab:per_domain_polybench

Caption: Per-category PolyBench metrics. Categories are inferred by keyword match on the trajectory prompt's Event description; other catches unmatched markets. Each cell reports Accuracy (%) / Return (%). Bold marks the best system per row on Accuracy.

| lrcccccc@ Domain | N | Sonnet | A-Evolve | Meta-Harness | Multi-agent | Adaptive | Full System |
| --- | --- | --- | --- | --- | --- | --- | --- |
| politics | 372 | 18.3/-5 | 11.3/+1 | 71.8/-2 | 87.9/-3 | 87.6/-2 | 87.1/-4 |
| sports | 1,120 | 23.8/+3 | 16.4/+8 | 55.8/+594 | 81.7/+659 | 80.0/+642 | 84.6/+596 |
| finance | 240 | 19.6/+2 | 18.3/+8 | 64.6/+7 | 81.7/+10 | 77.1/+8 | 80.4/+9 |
| crypto | 447 | 20.4/-3 | 17.7/+3 | 66.9/+2 | 89.3/+4 | 88.4/+4 | 90.8/+5 |
| entertainment | 218 | 22.0/-2 | 12.8/+1 | 64.7/+95 | 89.4/+105 | 87.6/+101 | 89.0/+83 |
| other | 2,678 | 22.6/+3 | 20.7/+9 | 40.8/+356 | 75.4/+394 | 72.3/+409 | 76.2/+374 |
| Overall | 5,075 | 22.2/+2 | 18.4/+7 | 50.8/+320 | 79.8/+351 | 77.4/+352 | 80.9/+330 |

## Table 9: tab:evolution_dynamics

Caption: Full-stream system contrast complementing the subset-based ablation in Figure fig:rq2_stats. No-evo is the no-evolution Sonnet baseline; Single is A-Evolve (single-agent evolver); Multi is the four-phase multi-agent evolver. The Peak column reports the cycle index at which the multi-agent run's cumulative mean was highest, and Cycles is the total number of evolution cycles.

| lrrrrrr@ Benchmark | Metric | No-evo | Single | Multi | Peak | Cycles |
| --- | --- | --- | --- | --- | --- | --- |
| PolyBench | Acc | 22.2 | 18.4 | 79.8 | 22 | 51 |
| CTF-Dojo | Pass@1 | 37.2 | 45.2 | 47.9 | 1 | 14 |
| FutureX | Pass@1 | 31.0 | 47.5 | 49.5 | 10 | 26 |

## Table 10: tab:branch_perf

Caption: Per-branch routing volume on the RQ4 navigation analysis subsets (:navloss). Each row reports tasks the LLM router actually sent to that branch under the Adapt condition, alongside the resulting Pass@1 (CTF-Dojo, FutureX) or HitRate among traded markets (PolyBench). On these subsets the router never invoked the main fallback that its prompt allows (Appendix app:prompt-router).

| llrr@ Bench | Branch | N routed | Pass / HitRate (%) |
| --- | --- | --- | --- |
| 4@lCTF-Dojo (60 tasks across 3 batches; per-task best-of-7 venues) |  |  |  |
| branch/crypto | 20 | 50.0 |  |
| branch/rev | 20 | 30.0 |  |
| branch/pwn | 12 | 0.0 |  |
| branch/misc | 4 | 75.0 |  |
| branch/web | 2 | 0.0 |  |
| branch/forensics | 2 | 50.0 |  |
| 4@lPolyBench (100 tasks across 4 batches; per-task best-of-5 venues) |  |  |  |
| branch/sports | 71 | 67.6 |  |
| branch/finance | 14 | 21.4 |  |
| branch/culture | 12 | 33.3 |  |
| branch/politics-world | 3 | 66.7 |  |
| 4@lFutureX (80 tasks across 3 batches; per-task best-of-5 venues) |  |  |  |
| branch/lvl1 | 28 | 53.6 |  |
| branch/lvl2 | 30 | 30.0 |  |
| branch/lvl3 | 8 | 0.0 |  |
| branch/lvl4 | 14 | 14.3 |  |

## Table 11: tab:routing_stats

Caption: Numeric companion to Figure fig:rq3_stats. Replay-based Oracle/Adapt/Naive/Worst comparison on the RQ4 subsets (:navloss). Oracle is the best venue per task; Adapt is the LLM router's choice; Naive is the fixed main venue (no branching); Worst is the worst venue per task. CTF-Dojo and FutureX use Pass@1 (%); PolyBench uses CWR (%). Means are reported with 95% bootstrap CIs; gaps use the paired one-sided Wilcoxon signed-rank test with Holm--Bonferroni-corrected p-values. The Oracle-Naive gap is the empirical adaptation loss L_adapt.

| lccc@ | CTF-Dojo | PolyBench | FutureX |
| --- | --- | --- | --- |
| (Pass%) | (CWR%) | (Pass%) |  |
| N tasks | 40 | 80 | 58 |
| Oracle | 55.0\,[40.0, 70.0] | +12.0\,[+2.1, +21.2] | 46.6\,[32.8, 60.3] |
| Adapt | 35.0\,[20.0, 50.0] | +5.9\,[-6.6, +17.4] | 34.5\,[22.4, 46.6] |
| Naive | 17.5\,[7.5, 30.0] | +3.2\,[-7.5, +13.0] | 39.7\,[27.6, 51.7] |
| Worst | 7.5\,[0.0, 17.5] | -10.6\,[-25.9, +4.0] | 22.4\,[12.1, 32.8] |
| L_adapt (Oracle-Naive) | +37.5 | +8.8 | +6.9 |
| Adapt-Naive | +17.5 | +2.7 | -5.2 |

## Table 12: tab:routing_per_batch

Caption: Per-batch breakdown of the Oracle/Adapt/Naive/Worst means on the RQ4 routing subsets. CTF-Dojo and FutureX use Pass@1 (%); PolyBench uses CWR (%). Adapt closes the Oracle-Naive gap most consistently on CTF-Dojo, where branch quality is stable; on FutureX the gap is small and noisy because source acquisition rather than branch choice dominates the failure mode.

| lrrrrrr@ Benchmark | Batch | N | Oracle | Adapt | Naive | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 2*CTF-Dojo (Pass%) | 2 | 20 | 45.0 | 30.0 | 15.0 | 10.0 |
| 3 | 20 | 65.0 | 40.0 | 20.0 | 5.0 |  |
| 4*PolyBench (CWR%) | 2 | 20 | +16.7 | +11.6 | +7.8 | -100.0 |
| 3 | 20 | +13.0 | +7.8 | +1.9 | -8.9 |  |
| 4 | 20 | +1.1 | +1.1 | +1.4 | -5.1 |  |
| 5 | 20 | +14.7 | +5.2 | +0.5 | -4.7 |  |
| 3*FutureX (Pass%) | 2 | 20 | 45.0 | 40.0 | 45.0 | 40.0 |
| 3 | 19 | 57.9 | 26.3 | 52.6 | 21.1 |  |
| 4 | 19 | 36.8 | 36.8 | 21.1 | 5.3 |  |

## Table 13: tab:hitl_log

Caption: Complete human-in-the-loop event log from the FutureX RQ5 run (Design C, 5 batches 20 tasks). P2 (cred.) is the research-phase credential hook; P3 (board) is the task-board steering hook. Two P2 events fire at cycle 1 to bootstrap the search pipeline; one substantive P3 event fires at cycle 3 to direct the evolver toward Western and Chinese specialty endpoints. The remaining P3 prompts return skip, matching the cheat-sheet protocol. API tokens supplied via Telegram are redacted.

| rllp0.220.34@ Cycle | Hook | Key | Trigger context | Human response |
| --- | --- | --- | --- | --- |
| 1 | P2 (cred.) | EXA_API_KEY | Phase-2 research needs Exa search API. | [REDACTED] (key supplied) |
| 1 | P2 (cred.) | SERPER_API_KEY | Phase-2 research needs Serper Google API. | [REDACTED] (key supplied) |
| 1 | P3 (board) | cycle1 | 7 tasks fail: solver has no web search tool. | skip |
| 2 | P3 (board) | cycle2 | 9 tasks fail: search-pipeline entry never invokes search functions. | skip |
| 3 | P3 (board) | cycle3 | deterministic fallback used; structured-data API gap. | ``Specialty data tasks need direct endpoint integrations beyond generic web search. Build skills/tools for Western (US equities OHLC, Box Office Mojo, ) and Chinese (Eastmoney secid, Maoyan film board, KolRank, ) endpoints.'' |
| 4 | P3 (board) | cycle4 | 17 tasks fail: deterministic fallback used. | skip |
| 5 | P3 (board) | cycle5 | 20 tasks fail: Chinese niche-ranking API absent. | skip |

## Table 14: tab:run_turns_time

Caption: Per-task solver turns and wall-clock seconds across systems and benchmarks. A turn is one tool call, including the final submit; a task that submits directly without other tool use therefore counts as 1. turns and sec are arithmetic means; median columns are added because both distributions are right-skewed on CTF-Dojo and FutureX. Wall-clock excludes orchestration overhead.

| l rrrr rrrr rrrr@ | 4cPolyBench | 4cCTF-Dojo | 4cFutureX |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (lr)2-5 (lr)6-9 (lr)10-13 System | turns | med | sec | med | turns | med | sec | med | turns | med | sec | med |
| Sonnet | 1.0 | 1.0 | 18.2 | 18.1 | 89.4 | 48.0 | 308.4 | 282.8 | 13.4 | 16.0 | 244.4 | 271.4 |
| A-Evolve | 1.7 | 2.0 | 17.0 | 15.7 | 17.8 | 10.0 | 156.7 | 76.9 | 16.9 | 6.0 | 98.8 | 41.9 |
| GEPA | 2.1 | 2.0 | 30.4 | 22.5 | 50.1 | 33.0 | 250.3 | 219.7 | 0.8 | 1.0 | 4.6 | 4.6 |
| Meta-Harness | 2.4 | 2.0 | 35.2 | 29.5 | 51.3 | 39.0 | 275.9 | 244.4 | 4.6 | 4.0 | 56.4 | 50.4 |
| Continual H. | 2.3 | 2.0 | 27.4 | 21.6 | 15.3 | 12.0 | 115.3 | 82.3 | 6.8 | 5.0 | 167.8 | 112.7 |
| SkillOS | 2.7 | 2.0 | 32.1 | 23.0 | 31.4 | 16.0 | 224.2 | 161.0 | 5.3 | 4.0 | 120.3 | 84.1 |
| OctoTools | 5.2 | 4.0 | 68.6 | 63.5 | 44.1 | 35.0 | 264.8 | 253.7 | 1.0 | 1.0 | 10.7 | 10.5 |
| Multi-agent | 5.7 | 6.0 | 55.3 | 48.7 | 40.1 | 35.0 | 293.1 | 245.2 | 11.3 | 3.0 | 86.4 | 36.2 |
| Adaptive | 2.8 | 3.0 | 43.7 | 38.3 | 29.7 | 25.0 | 281.9 | 238.1 | 10.1 | 8.0 | 74.5 | 55.7 |
| Full System | 5.4 | 6.0 | 42.2 | 40.5 | 36.4 | 42.0 | 290.9 | 255.9 | 4.2 | 4.0 | 47.6 | 38.6 |
