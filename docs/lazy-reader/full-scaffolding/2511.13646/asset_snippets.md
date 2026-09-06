# Paper Assets

## Figure 1: fig:result

Caption: Verified and results (single attempt w/o test-time scaling)

![Verified and results (single attempt w/o test-time scaling)](figures/sb_results.png)

## Figure 2: fig:overview

Caption: Overview of

![Overview of](figures/overview.png)

## Table 1: tab:verified

Caption: Result on Verified

| Tool | % Resolved | Avg. Cost |
| --- | --- | --- |
| 59.8% | 0.04 |  |
| 65.0% | 0.28 |  |
| 70.6% | 0.56 |  |
| -4* yang2024sweagent | 74.2% | 0.46 |
| 63.0% | 0.05 |  |
| 68.4% | 0.27 |  |
| 75.4% | 0.68 |  |
| -4* | 77.4% | 0.48 |

## Table 2: tab:verified_subset

Caption: Result on Verified-60

| Tool | % Resolved | Offline cost (hours) |  |
| --- | --- | --- | --- |
| 3*90 self- |  |  |  |
| improving |  |  |  |
| agents | robeyns2025sica | 50.0% | infinite loop |
| 2-5 | zhang2025darwin | 53.3% | 1231 |
| 2-5 | wang2025huxley | 56.7% | 512 |
| 65.0% | 0 |  |  |

## Table 3: tab:pro

Caption: Result on

| Tool | % Resolved | Avg. Cost |
| --- | --- | --- |
| yang2024sweagent | 43.6% | - |
| 45.8% | 0.73 |  |

## Figure 3: fig:tool_verified

Caption: Tools in Verified

![Tools in Verified](figures/tsne_verified_claude_tool_use.png)

## Table 4: tab:ablation_setup

Caption: Ablation result across setups on subset of Verified problems

| Approach | % Resolved | # of tools created |
| --- | --- | --- |
| w/o tool creation | 62.0% | 0.00 |
| w/o reflection | 64.0% | 2.92 |
| 76.0% | 3.28 |  |

## Table 5: tab:ablation_llm

Caption: Ablation result across different backends on subset of Verified problems

| 44.0% | 14.0% red(-68.2%) |
| --- | --- |
| 60.0% | 58.0% red(-3.3%) |
| 60.0% | 68.0% 009933(13.3%) |
| 46.0% | 50.0% 009933(8.7%) |
| 58.0% | 64.0% 009933(10.3%) |
| 62.0% | 76.0% 009933(22.6%) |

## Table 6: tab:multilingual

Caption: Result on a subset of problems

| Tool | % Resolved | Avg. Cost |
| --- | --- | --- |
| yang2024sweagent | 40.0% | 0.59 |
| 46.0% | 0.66 |  |

## Figure 4: fig:tool_verified_repo

Caption: Tools in Verified

![Tools in Verified](figures/tsne_verified_claude_repo.png)
