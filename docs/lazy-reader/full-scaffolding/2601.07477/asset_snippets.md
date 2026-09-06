# Paper Assets

## Figure 1: fig:catchy

Caption: Block-level Judge guides agentic workflow optimization by identifying the most problematic block in failed executions.

![Block-level Judge guides agentic workflow optimization by identifying the most problematic block in failed executions.](figures/catchy2.png)

## Figure 2: fig:logic_block

Caption: The illustration of logic blocks.

![The illustration of logic blocks.](figures/logic_block.png)

## Figure 3: fig:judgeflow

Caption: The main pipeline of

![The main pipeline of](figures/judge4.png)

## Table 1: tab:full_width

Caption: width=0.8

| Method | GSM8K | MATH | MBPP | HumanEval | Avg. |
| --- | --- | --- | --- | --- | --- |
| 6cSingle-agent System |  |  |  |  |  |
| IO | 87.8 | 48.6 | 73.9 | 87.0 | 74.3 |
| CoT wei2023chainofthoughtpromptingelicitsreasoning | 87.0 | 48.8 | 74.2 | 88.6 | 74.7 |
| CoT SC wang2023selfconsistencyimproveschainthought | 86.9 | 50.4 | 73.3 | 91.6 | 75.6 |
| 6cHand-crafted Multi-agent System |  |  |  |  |  |
| SELF-REFINE madaan_self-refine_nodate | 85.5 | 46.1 | 71.8 | 87.8 | 72.8 |
| LLM-Debate du_improving_2023 | 89.5 | 48.6 | 70.3 | 88.8 | 74.3 |
| LLM-Blender jiang2023llmblenderensemblinglargelanguage | 88.4 | 46.9 | 77.1 | 88.7 | 75.3 |
| DyLAN liu_dynamic_2024 | 90.0 | 48.5 | 77.3 | 90.4 | 76.6 |
| 6cAutonomous Multi-agent System |  |  |  |  |  |
| GPTSwarm zhuge_language_2024 | 89.1 | 47.9 | 77.4 | 89.3 | 75.9 |
| ADAS hu_automated_2025 | 88.4 | 43.2 | 77.1 | 84.2 | 73.2 |
| AFlow zhang_aflow_2025 | 90.1 | 52.8 | 81.7 | 90.1 | 78.7 |
| MaAS zhang_multi-agent_2025 | 91.5 | 52.2 | 82.2 | 91.6 | 79.4 |
| MermaidFlow zheng_mermaidflow_2025 | 92.4 | 55.4 | 82.3 | 92.9 | 80.8 |
| (Ours) | 93.0 | 58.5 | 83.8 | 93.4 | 82.2 |

## Figure 4: fig:aime2025

Caption: Performance on AIME 2025. The results are evaluated averaged over five independent runs. We use gpt-4.1-mini in the experiments.

No embeddable figure file was copied.

## Figure 5: fig:ab1

Caption: The optimal workflow found by JudgeFlow on the MBPP dataset.

![The optimal workflow found by JudgeFlow on the MBPP dataset.](figures/MBPP-result.png)

## Figure 6: fig:ab2

Caption: Training and testing curves of JudgeFlow and AFlow on the MBPP dataset.

No embeddable figure file was copied.

## Figure 7: fig:case-study

Caption: The illustration of the case study in the GSM8K dataset.

![The illustration of the case study in the GSM8K dataset.](figures/case_study2.png)

## Table 2: tab:ablation

Caption: width=0.8

| Models | Score |
| --- | --- |
| GPT-4o-mini | 83.8 |
| GPT-4o | 84.5 |
| Gemini-2.5-flash | 84.4 |

## Table 3: tab:cross-generalization

Caption: Cross-Benchmark Transfer Performance

| AFlow |  |  |
| --- | --- | --- |
| MATH GSM8K | 91.95 | 92.89 |
| MBPP HumanEval | 90.84 | 93.89 |
