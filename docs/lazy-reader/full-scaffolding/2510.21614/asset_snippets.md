# Paper Assets

## Figure 1: unlabeled

Caption: No caption extracted.

![Figure 1](figures/pull_force.png)

## Figure 2: fig:intro

Caption: (Left) Weak correlation between the guidance metrics of other methods (based on benchmark performance) and long-term self improvement; HGM mitigates this mismatch by leveraging clade-level metaproductivity. (Right) On SWE-bench Verified, HGM achieves higher accuracy with 2.38 time less allocated CPU-hours. Together, the results indicate the practical advantage of approximating G\"odel Machines with long-term self-improvement estimates. Note that SICA encountered repeated errors after consuming 45% of its budget, preventing any further self-modifications.

![(Left) Weak correlation between the guidance metrics of other methods (based on benchmark performance) and long-term self improvement; HGM mitigates this mismatch by leveraging clade-level metaproductivity. (Right) On SWE-bench Verified, HGM achieves higher accuracy with 2.38 time less allocated CPU-hours. Together, the results indicate the practical advantage of approximating G\"odel Machines with long-term self-improvement estimates. Note that SICA encountered repeated errors after consuming 45% of its budget, preventing any further self-modifications.](figures/intro_figure_combined_wider.png)

## Table 1: tab:vs-corr

Caption: table

```tex
[t!]
  \centering
  \captionof{table}{
  \textbf{Clade-Metaproductivity: Empirical vs. Estimation Correlation}. We report the Pearson correlations between the empirical $\mathrm{CMP}s$ and the estimates from DGM, SICA, and HGM on SWE-Verified-60 and Polyglot. For the weighted correlations, each prediction is weighted by the number of evaluations it has accessed.
  }\label{tab:vs-corr}
  \renewcommand\arraystretch{1.2} 
  \resizebox{0.9\linewidth}{!}{
    \begin{tabularx}{\linewidth}{
      l|
          >{\columncolor{lightwheat!70}}Y
          >{\columncolor{skyblue!20}}Y|
          >{\columncolor{lightwheat!70}}Y
          >{\columncolor{skyblue!20}}Y
    } 
      \toprule
      \multirow{2}{*}{\textbf{Estimates}} 
        & \multicolumn{2}{c|}{\textbf{SWE-Verified-60}} 
        & \multicolumn{2}{c}{\textbf{Polyglot}} \\
      \cmidrule(lr){2-3}\cmidrule(l){4-5}
        & \multicolumn{1}{c}{\textbf{Weighted}} & \multicolumn{1}{c}{\textbf{Un-weighted}} 
        & \multicolumn{1}{c}{\textbf{Weighted}} & \multicolumn{1}{c}{\textbf{Un-weighted}} \\ 
      \midrule
      SICA & 0.444    & 0.444      & 0.274     & 0.274 \\
      DGM  & 0.285    & 0.406      & 0.383     & 0.357 \\ 
      \hline
      \textbf{HGM (Ours)} & \textbf{0.778} & \textbf{0.512}  & \textbf{0.626} & \textbf{0.873}  \\
      \hline
      \bottomrule
    \end{tabularx}
  }
```

## Table 2: tab:self-improv-compare

Caption: table

```tex
[t!]
  \centering
  \captionof{table}{
  \textbf{Self-Improving Capability Comparison}. We report the task performance (in accuracy) of each method’s best-belief agent and the allocated CPU-hours time required for 800 evaluations. Superscripted accuracies with ``$+$'' indicate performance gains over their respective initial agents. 
  }
  \label{tab:self-improv-compare}
  \renewcommand\arraystretch{1.2} 
  \resizebox{0.9\linewidth}{!}{
    \begin{tabularx}{\linewidth}{
      l|
          >{\columncolor{lightwheat!70}}Y
          >{\columncolor{skyblue!20}}Y|
          >{\columncolor{lightwheat!70}}Y
          >{\columncolor{skyblue!20}}Y
    } 
      \toprule
      \multirow{2}{*}{\textbf{Best-belief Agent of}} 
        & \multicolumn{2}{c|}{\textbf{SWE-Verified-60}} 
        & \multicolumn{2}{c}{\textbf{Polyglot}} \\
      \cmidrule(lr){2-3}\cmidrule(l){4-5}
        & \multicolumn{1}{c}{\textbf{Acc. (\%)$\uparrow$}} & \multicolumn{1}{c}{\textbf{Time (hours)$\downarrow$}} 
        & \multicolumn{1}{c}{\textbf{Acc. (\%)$\uparrow$}} & \multicolumn{1}{c}{\textbf{Time (hours)$\downarrow$}} \\
      \midrule
      SICA & 50.0$^{+10}$     & infinite loop       & 25.4$^{+5.1}$     & 572 \\
      DGM  & 53.3$^{+13.3}$   & 1231      & 27.1$^{+6.8}$     & 2385 \\ 
      \hline
      \textbf{HGM (Ours)} & \textbf{56.7}$^{+16.7}$ & \textbf{517} & \textbf{30.5}$^{+10.2}$ & \textbf{347} \\
      \hline
      \bottomrule
    \end{tabularx}
  }
```

## Table 3: tab:vs-human-lite

Caption: table

```tex
[t!]
  \centering
  \captionof{table}{
  \textbf{Generalization on SWE-Lite: HGM's Best-belief SWE-Verified Agent.} We report the accuracy of HGM’s best-belief SWE-Verified agent on SWE-Lite under two settings: filtered (excluding tasks overlapping with SWE-Verified) and standard (the official leaderboard setting used for evaluating human-designed agents)).
  }\label{tab:vs-human-lite}
  \renewcommand\arraystretch{1.2}
  \resizebox{0.9\linewidth}{!}{
  \begin{tabularx}{\linewidth}{
      l|
          >{\columncolor{lightwheat!70}}Y|
          >{\columncolor{skyblue!20}}Y
    } 
    \toprule
    \multicolumn{1}{c|}{\textbf{Coding Agents}}   & \multicolumn{1}{c|}{\textbf{SWE-Lite Filtered (\%)}} & \multicolumn{1}{c}{\textbf{SWE-Lite Standard (\%)}} \\
    \hline
      HGM Initial Ancestor    & 34.8   & 44.0 \\
      SWE-agent+GPT-5-mini    & 39.6   & 47.6 \\
    \hline
     \textbf{HGM's Best-belief SWE-Verified Agent} & \textbf{40.1}   & \textbf{49.0} \\ 
    \hline    
    \bottomrule
  \end{tabularx}
}
```

## Table 4: tab:gpt5-vs-human-lite

Caption: table

```tex
[t!]
  \centering
  \captionof{table}{
  \textbf{Transfer to different LLMs on SWE-Lite: HGM's Best-belief SWE-Verified Agent.} Similarly, We report the accuracy of HGM’s best-belief SWE-Verified (optimized with GPT-5-mini) agent on SWE-Lite (evaluated with GPT-5) under two settings: filtered (excluding tasks overlapping with SWE-Verified) and standard (the official leaderboard setting used for evaluating human-designed agents).
  }\label{tab:gpt5-vs-human-lite}
  \renewcommand\arraystretch{2}
  \resizebox{0.9\linewidth}{!}{
  \begin{tabularx}{\linewidth}{
      l|
          >{\columncolor{lightwheat!70}}Y|
          >{\columncolor{skyblue!20}}Y
    } 
    \toprule
    \multicolumn{1}{c|}{\textbf{Coding Agents}}   & \multicolumn{1}{c|}{\textbf{SWE-Lite Filtered (\%)}} & \multicolumn{1}{c}{\textbf{SWE-Lite Standard (\%)}} \\
    \hline
    SWE-agent (Best on the LB)    & \textbf{48.3}   & 56.7 \\
    \hline
     \parbox[c][5ex][c]{5.9cm}{
      \textbf{HGM's Best-belief SWE-Verified Agent + GPT-5} 
     }
     & 47.8   & \textbf{57} \\ 
     \hline
    \bottomrule
  \end{tabularx}
}
```

## Table 5: tab:structured_policy

Caption: Comparison of structured policies across self-improving methods. Each method is described by three subpolicies: Selection Policy, Expansion Policy, and Evaluation Policy.

```tex
[htbp]
\centering
\renewcommand{\arraystretch}{1.25}
\begin{tabularx}{\textwidth}{p{2.2cm} X X X}

\toprule
\textbf{Subpolicy} & \textbf{SICA} & \textbf{DGM} & \textbf{HGM (Ours)} \\
\midrule
\textbf{Selection Policy} 
& Alternates between modification and evaluation. 
& Alternates between modification and evaluation. 
& Adaptive choice between modification and evaluation. \\
\midrule
\textbf{Expansion Policy} 
& Greedily selects the agent with the best performance up to this point and modifies it with the entire history accessible to the agent. 
& Selects the node probabilistically based on the evaluation metric and the number of children of the agents. 
& Selects the node based on the statistics of the \emph{clade} stemming from a given node.\\
\midrule
\textbf{Evaluation Policy} 
& Evaluates the most recently created agent on the entire evaluation dataset. 
& Progressively evaluates the last created agent on subsets of the dataset, expanding if results are promising. 
& Selects the agent based on the statistics and evaluates it on a single task. \\
\bottomrule
\end{tabularx}
\caption{Comparison of structured policies across self-improving methods. Each method is described by three subpolicies: Selection Policy, Expansion Policy, and Evaluation Policy. }
\label{tab:structured_policy}
```
