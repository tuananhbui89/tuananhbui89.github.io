# Paper Assets

## Figure 1: fig:system-overview

Caption: RoboPhD system overview. Simple inputs (a naive agent, an evolution strategy, and training data) feed into an ELO-based evolutionary loop that produces a strong, production-ready agent for deployment.

No embeddable figure file was copied.

## Figure 2: fig:bird-flowchart

Caption: The RoboPhD evolutionary cycle. The Database Analysis Tool (gray) is a deterministic Python script, not an LLM call, making offline analysis fast, very cheap, and reproducible.

No embeddable figure file was copied.

## Table 1: tab:main-results

Caption: Results on BIRD dev set

| 2cOpus-4.5 | 2cSonnet-4.5 | 2cHaiku-4.5 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| (lr)2-3 (lr)4-5 (lr)6-7 | Accuracy | Cost/Query | Accuracy | Cost/Query | Accuracy | Cost/Query |
| Naive | 69.0% | 1.61 | 65.7% | 0.56 | 57.2% | 0.34 |
| Best Evolved | 71.3% | 3.13 | 69.2% | 0.87 | 66.1% | 0.51 |
| +2.3 | +3.5 | +8.9 |  |  |  |  |

## Table 2: tab:test-results

Caption: Official BIRD test set results (Opus 4.5)

| Total | Simple | Moderate | Challenging |  |
| --- | --- | --- | --- | --- |
| Naive | 72.16% | 81.35% | 68.65% | 48.42% |
| Best Evolved | 73.67% | 81.03% | 70.45% | 55.44% |
| +1.51 | -0.32 | +1.80 | +7.02 |  |

## Table 3: tab:size-adaptive

Caption: Size-adaptive feature matrix. The agent gracefully degrades analysis depth for larger databases to prevent overflowing the 200K token Claude context limit while maintaining comprehensive analysis for typical BIRD databases.

| Feature | Small | Medium | Large | Ultra |
| --- | --- | --- | --- | --- |
| (150 cols) | (300 cols) | (400 cols) | (>400 cols) |  |
| Sample values/column | 10 | 5 | 3 | 1 |
| Enum value limit | All | 15 | 5 | 0 |
| Semantic patterns | Full | Essential | Skip | Skip |
| Cross-table validation | Full | Critical | Skip | Skip |
