# Paper Assets

## Figure 1: fig:hcl_concept

Caption: The shift in the object of continual learning. Model-centric methods update model parameters over sequential experience. HCL instead updates harness state around a frozen foundation model. In both settings, adaptation can improve new behavior while interfering with behavior acquired earlier.

![The shift in the object of continual learning. Model-centric methods update model parameters over sequential experience. HCL instead updates harness state around a frozen foundation model. In both settings, adaptation can improve new behavior while interfering with behavior acquired earlier.](figures/intro.png)

## Table 1: tab:hcl_state_overview

Caption: Execution functions and updatable contents of the four jointly versioned components in the deployed harness H_n.

| p0.18 >0.33 >0.39 Component | Function during execution | Contents updated in HCL |
| --- | --- | --- |
| Task Interface I_n | Transforms raw interactions into structured representations. | Prompts, task templates, and parsing and normalization rules. |
| [3pt] Experience Memory M_n | Provides concrete interactions and abstract guidance for reuse. | Raw interaction records and LLM-generated Abstract Memory entries. |
| [3pt] Capability Map C_n | Provides external operations and reusable inner skills. | Inner skills extracted from Abstract Memory. |
| [3pt] Adaptive Router R_n | Selects and organizes memory and capabilities. | Routing prompts, selection criteria, and workflow templates. |

## Figure 2: fig:hcl_framework

Caption: Overview of the HCL framework. The deployed harness H_n supports the execution path from raw interaction u_n to outcome y_n. When post-execution feedback is available, the Continual Optimizer proposes a candidate harness H_n+1, and the Continual Evaluator accepts or rejects it based on current improvement, historical retention, and validity.

![Overview of the HCL framework. The deployed harness H_n supports the execution path from raw interaction u_n to outcome y_n. When post-execution feedback is available, the Continual Optimizer proposes a candidate harness H_n+1, and the Continual Evaluator accepts or rejects it based on current improvement, historical retention, and validity.](figures/framework.png)

## Table 2: tab:alfworld

Caption: Final performance and harness-level forgetting on ALFWorld with Qwen3.5-9B as the frozen foundation model. The best and second-best results in each metric column are marked in bold and underlined, respectively.

| Method | Pick | Look | Clean | Heat | Cool | Two-object | Final Avg. \(\) | Avg. Fgt. \(\) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static Harness | 95.80 | 66.70 | 25.80 | 26.10 | 9.50 | 58.80 | 47.12 | -- |
| RAG Baseline | 95.80 | 83.30 | 41.90 | 39.10 | 14.30 | 58.80 | 55.56 | 1.74 |
| MemP memp | 95.80 | 83.30 | 48.40 | 34.80 | 9.50 | 47.10 | 53.15 | 5.18 |
| MemRL zhang2026memrl | 87.50 | 66.70 | 29.00 | 60.90 | 23.80 | 41.20 | 51.51 | 5.64 |
| Stability-HCL (Ours) | 100.00 | 83.30 | 51.60 | 30.40 | 28.60 | 76.50 | 61.74 | 2.64 |
| Plasticity-HCL (Ours) | 100.00 | 77.80 | 41.90 | 39.10 | 19.00 | 100.00 | 62.98 | 10.94 |

## Figure 3: fig:minecraft_evolution

Caption: Curriculum progression and execution efficiency. (a) HCL completes all 50 tasks, while the Static Harness plateaus at 15. (b) Cumulative environment actions over the 50-task curriculum: HCL uses 83, versus 88 for MemRL and 91 for MemP; lower is more efficient.

![Curriculum progression and execution efficiency. (a) HCL completes all 50 tasks, while the Static Harness plateaus at 15. (b) Cumulative environment actions over the 50-task curriculum: HCL uses 83, versus 88 for MemRL and 91 for MemP; lower is more efficient.](figures/curriculum_comparison.png)

## Table 3: tab:text_main_results

Caption: Final performance after the four-task textual-reasoning stream with DeepSeek-V4-Flash as the frozen foundation model. The Zero-shot baseline evaluates each task independently without sequential harness updates. The best and second-best results in each metric column are marked in bold and underlined, respectively.

| Method | MuSiQue | ProofWriter | GSM8K | HotpotQA | Final Avg. | Avg. Fgt. |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash Zero-shot | 35.00 | 42.80 | 49.40 | 54.80 | 45.50 | -- |
| Stability-HCL (Ours) | 27.60 | 73.00 | 50.40 | 57.80 | 52.20 | 0.00 |
| Plasticity-HCL (Ours) | 29.00 | 77.00 | 92.00 | 60.80 | 64.70 | 0.07 |

## Table 4: tab:multimodal_results

Caption: Final performance after the four-task multimodal-perception stream with Qwen3.6-27B as the frozen foundation model. The Zero-shot baseline evaluates each task independently without sequential harness updates. The best and second-best results in each metric column are marked in bold and underlined, respectively.

| Method | Detection | Caption | Grounding | VQAv2 | Final Avg. | Avg. Fgt. |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B Zero-shot | 4.27 | 25.47 | 43.00 | 84.87 | 39.40 | -- |
| DGG li2026dgg | 29.58 | 29.77 | 48.96 | 62.60 | 42.73 | 0.26 |
| Plasticity-HCL (Ours) | 64.14 | 37.31 | 90.60 | 79.80 | 67.96 | 0.81 |
| Stability-HCL (Ours) | 65.34 | 39.41 | 91.60 | 79.33 | 68.92 | 0.22 |

## Table 5: tab:text_threshold

Caption: Performance under different fixed values of b, where B_n b within each run. All other experimental conditions are held constant. The best and second-best results in each metric column are marked in bold and underlined, respectively.

| Historical-loss tolerance b | MuSiQue | ProofWriter | GSM8K | HotpotQA | Final Avg. | Avg. Fgt. |
| --- | --- | --- | --- | --- | --- | --- |
| b=0 | 27.83 | 73.33 | 84.33 | 59.50 | 61.25 | 0.39 |
| b=1 | 24.83 | 77.50 | 92.33 | 59.17 | 63.46 | 1.22 |
| b=3 | 26.83 | 79.83 | 83.00 | 58.50 | 62.04 | 2.00 |
| b= | 28.33 | 71.00 | 82.00 | 59.17 | 60.13 | 3.45 |

## Figure 4: fig:textual_forgetting

Caption: Textual reasoning under different fixed values of b.

![Textual reasoning under different fixed values of b.](figures/textual_forgetting_trajectory_styled.png)

## Table 6: tab:component_ablation

Caption: Component ablation on the controlled multimodal stream. I, M, C, and R denote the Task Interface, Experience Memory, Capability Map, and Adaptive Router. A check mark indicates that the component is updated, while a cross indicates that its update is disabled. The best and second-best results in each metric column are marked in bold and underlined, respectively.

| Component | I | M | C | R | Final Avg. | Avg. Fgt. |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | -- | -- | -- | -- | 34.84 | -- |
| w/o Interface update | 62.37 | 0.11 |  |  |  |  |
| w/o Memory update | 62.28 | 0.83 |  |  |  |  |
| w/o Capability update | 63.12 | 0.06 |  |  |  |  |
| w/o Router update | 62.77 | 0.14 |  |  |  |  |
| Full HCL | 63.41 | 0.45 |  |  |  |  |

## Table 7: tab:state_access_boundaries

Caption: Access and update boundaries of the deployed harness and anchor set.

| p0.20 >0.40 >0.30 Artifact | Execution and candidate-generation access | Update boundary |
| --- | --- | --- |
| Task Interface I_n | Constructs i_n; the Optimizer may revise prompts, templates, and parsing or normalization rules. | Changes enter H_n only with a committed candidate. |
| Raw and Abstract Memory M_n^raw,M_n^abs | Supplies records and guidance to the Router; the Optimizer may add raw records or revise abstract entries. | Changes enter H_n only with a committed candidate. |
| Capability Map C_n | Supplies capabilities to the Router; the Optimizer may add or revise internal skills. | Changes enter H_n only with a committed candidate. |
| Adaptive Router R_n | Constructs z_n; the Optimizer may revise routing prompts, selection criteria, or workflow templates. | Changes enter H_n only with a committed candidate. |
| Anchor Set A_n | Used only by the Evaluator; unavailable to execution and candidate generation. | Updated at the end of each task and then fixed during candidate generation and evaluation for the next task. |

## Table 8: tab:reproducibility_settings

Caption: Experimental settings. Counts are per task or category unless a stream total is stated.

| p0.22 p0.22 p0.31 Experiment | Stream and frozen model | Adaptation/evaluator data | Final reporting |
| --- | --- | --- | --- |
| ALFWorld main | Six categories in the order Pick-and-Place, Look-in-Light, Clean, Heat, Cool, and Two-object; frozen Qwen3.5-9B. | 10 training episodes per category, with at most 50 interaction steps per episode. Evaluation on all observed categories after each stage. | Final success on 134 official evaluation episodes, category macro-average, and average forgetting over the first five categories. |
| Minecraft main | 50 tasks covering collection, crafting, mining, tool use, placement, smelting, and multi-step dependencies; frozen Qwen3.6-27B. | Sequential environment feedback, with retained skill tests as historical anchors. | Cumulative task completion, recovery events, and validated skill changes. Completed tasks are not systematically replayed after every update. |
| Textual main | MuSiQue ProofWriter GSM8K HotpotQA; frozen DeepSeek-V4-Flash. | 250 adaptation and 50 validation examples per task. | 500 test examples per task. Final task scores, average performance, and forgetting. |
| Multimodal main | COCO detection COCO captioning RefCOCO grounding VQAv2; frozen Qwen3.6-27B. | 250 adaptation and 50 validation examples per task. | 500 test examples per task. Final task scores, average performance, and forgetting. |
| Textual budget sweep | The same textual order; frozen DeepSeek-V4-Flash. | 300 adaptation and 80 validation examples per task, with 80 anchors retained for each earlier task. | 600 test examples per task. Each profile receives 40 proposals, with ten at each task stage. |

## Table 9: tab:component_ablation_config

Caption: Update scope of the component-ablation variants. A permits persistent updates, while keeps the component fixed.

| Method | I | M | C | R | Fixed contents |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | -- | -- | -- | -- | No structured HCL harness or persistent updates. |
| Full HCL | None. |  |  |  |  |
| w/o Interface update | Prompts, templates, parsing, and normalization rules. |  |  |  |  |
| w/o Memory update | Raw and Abstract Memory entries. |  |  |  |  |
| w/o Capability update | Reusable skills. |  |  |  |  |
| w/o Router update | Routing prompts, selection criteria, and workflow templates. |  |  |  |  |

## Table 10: tab:component_ablation_full

Caption: Full component-ablation results on the controlled multimodal stream. ``Committed'' counts candidate updates entering the persistent harness.

| Method | Detection | Caption | Grounding | VQAv2 | Final Avg. | Avg. Fgt. | Committed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 35.11 | 22.98 | 0.00 | 81.27 | 34.84 | -- | -- |
| Full HCL | 53.07 | 36.09 | 87.60 | 76.87 | 63.41 | 0.45 | 18 |
| w/o Interface update | 53.45 | 33.56 | 87.80 | 74.67 | 62.37 | 0.11 | 24 |
| w/o Memory update | 55.50 | 28.95 | 88.00 | 76.67 | 62.28 | 0.83 | 46 |
| w/o Capability update | 55.11 | 34.16 | 86.40 | 76.80 | 63.12 | 0.06 | 16 |
| w/o Router update | 53.59 | 36.68 | 87.40 | 73.40 | 62.77 | 0.14 | 4 |

## Table 11: tab:anchor_criteria_text

Caption: Anchor success criteria for textual reasoning.

| p0.71 Task | q(H,a)=1 when |
| --- | --- |
| MuSiQue / HotpotQA | The normalized predicted short answer exactly matches an accepted reference answer. |
| ProofWriter | The parsed entailment label exactly matches the gold label and the output schema is valid. |
| GSM8K | The parsed final numeric value equals the gold value after comma and unit normalization. |

## Table 12: tab:anchor_criteria_multimodal

Caption: Anchor success criteria for multimodal perception.

| p0.71 Task | q(H,a)=1 when |
| --- | --- |
| COCO detection | For the queried annotated instance, the predicted category is correct, the matched bounding box has IoU 0.5, and the box schema is valid. |
| COCO captioning | Sentence-level CIDEr against the reference captions is at least 0.5 on the normalized [0,1] scale, and the caption schema is valid. |
| RefCOCO grounding | The predicted box is valid and has IoU 0.5 with the referred-object box. |
| VQAv2 | The standard VQA consensus score is 1.0 after answer normalization. |

## Table 13: tab:anchor_criteria_interactive

Caption: Anchor success criteria for interactive environments.

| p0.71 Environment | q(H,a)=1 when |
| --- | --- |
| ALFWorld | The environment's specified goal predicate is true within the 50-step limit under a valid action sequence. |
| Minecraft | The retained test for the corresponding skill reaches its predefined inventory or world-state predicate through a valid action sequence. |
