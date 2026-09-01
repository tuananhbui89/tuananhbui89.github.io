# Paper Assets

## Figure 1: fig:mainfig

Caption: Overview of self-improvement paradigms for modern AI agents. We categorize existing methods into two primary pathways according to what is modified. The first pathway is Foundation Model Improvement, where the model parameters are updated from _t to _t+1 using intrinsic generative demonstrations D_t, intrinsic evaluative feedback e_t, or extrinsic exploratory experience _t. The second pathway is Scaffolding Improvement, where the operational scaffold is updated from _t to _t+1 through non-parametric changes. Across scaffold components, a generic update signal S_t is instantiated to drive improvements in prompts p_t, memory m_t, tools T_t, or the full scaffolding _t.

![Overview of self-improvement paradigms for modern AI agents. We categorize existing methods into two primary pathways according to what is modified. The first pathway is Foundation Model Improvement, where the model parameters are updated from _t to _t+1 using intrinsic generative demonstrations D_t, intrinsic evaluative feedback e_t, or extrinsic exploratory experience _t. The second pathway is Scaffolding Improvement, where the operational scaffold is updated from _t to _t+1 through non-parametric changes. Across scaffold components, a generic update signal S_t is instantiated to drive improvements in prompts p_t, memory m_t, tools T_t, or the full scaffolding _t.](figures/fig_main.png)

## Figure 2: fig:agent_timeline

Caption: Timeline and taxonomy of self-improvement in foundation-model-based agents (2023–2026). Representative works are positioned by publication year. The left lane denotes foundation-model improvement (), while the right lane denotes scaffolding improvement (). The AGI signpost highlights the field’s long-term aspiration toward increasingly general agentic intelligence.

![Timeline and taxonomy of self-improvement in foundation-model-based agents (2023–2026). Representative works are positioned by publication year. The left lane denotes foundation-model improvement (), while the right lane denotes scaffolding improvement (). The AGI signpost highlights the field’s long-term aspiration toward increasingly general agentic intelligence.](figures/agent_timeline.png)

## Figure 3: fig:self_improving_taxonomy

Caption: A unified taxonomy of self-improving agents spanning foundation-model updates, scaffold updates, and evaluation benchmarks.

No embeddable figure file was copied.

## Figure 4: fig:timeline

Caption: A timeline of theoretical roots and idealized models for self-improving agents, from the late 1790s to the present.

![A timeline of theoretical roots and idealized models for self-improving agents, from the late 1790s to the present.](figures/timeline.png)

## Figure 5: fig:agent_component

Caption: Schematic of an FM-based agent under our formalism.

![Schematic of an FM-based agent under our formalism.](figures/agent_component.png)

## Figure 6: fig:FMI

Caption: Overview of foundation model improvement under agent-induced learning signals. The agent improves the foundation model by generating intrinsic demonstrations, producing intrinsic evaluative feedback, or collecting extrinsic exploratory experience, each forming a distinct parameter-update loop.

![Overview of foundation model improvement under agent-induced learning signals. The agent improves the foundation model by generating intrinsic demonstrations, producing intrinsic evaluative feedback, or collecting extrinsic exploratory experience, each forming a distinct parameter-update loop.](figures/fig-FMI.png)

## Figure 7: fig:prompt

Caption: Prompt refinement as a self-improvement loop and its four paradigms, organized by the learning signal S_t.

![Prompt refinement as a self-improvement loop and its four paradigms, organized by the learning signal S_t.](figures/fig-prompt.png)

## Table 1: tab:prompt_optimization_paradigms

Caption: Comparison of prompt optimization paradigms in prompt-based self-improvement. As learning signals become more structured and informative, optimization becomes less heuristic and more automated. 1: Scalar-feedback optimization; 2: Qualitative-feedback refinement; 3: Population-based evolution; 4: Textual-gradient optimization.

```tex
[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.30}

\caption{Comparison of prompt optimization paradigms in prompt-based self-improvement.
As learning signals become more structured and informative, optimization becomes less heuristic and more automated.
\textbf{\textcircled{1}}: Scalar-feedback optimization; \textbf{\textcircled{2}}: Qualitative-feedback refinement; \textbf{\textcircled{3}}: Population-based evolution; \textbf{\textcircled{4}}: Textual-gradient optimization.}
\vspace{0.4em}


\rowcolors{2}{lightBlue!18}{white}

\begin{tabularx}{\textwidth}{
>{\raggedright\arraybackslash}p{0.62cm} 
>{\raggedright\arraybackslash}p{1.10cm} 
>{\raggedright\arraybackslash}p{1.65cm}
>{\raggedright\arraybackslash}p{4.05cm} 
>{\raggedright\arraybackslash}X         
>{\raggedright\arraybackslash}X         
}
\toprule
\textbf{ID} &
\textbf{Signal $\mathcal{S}_t$} &
\textbf{Objective} &
\textbf{Representative Systems} &
\textbf{Advantages} &
\textbf{Limitations} \\
\midrule


\renewcommand{\arraystretch}{2.10}


\textbf{\fcirc{Blue}{1}}\;\prog &
\cellcolor{lightBlue!12}\makecell[l]{Scalar\\score} &
\cellcolor{lightBlue!8}$\arg\max_{p\in\mathcal{P}} f(p)$ &
\cellcolor{lightBlue!6}\makecell[l]{RLPrompt~\citep{deng2022rlprompt}\\
BBT~\citep{sun2022black}\\
APE~\citep{zhou2022large}\\
OPRO~\citep{yang2023large}\\
Dspy~\citep{khattab2023dspy}} &
\makecell[l]{\posbar \textcolor{green!55!black}{\bfseries +}\, Model-agnostic\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Simple to deploy\\
\posbar \textcolor{green!55!black}{\bfseries +}\, No internal access} &
\makecell[l]{\negbar \textcolor{red!70!black}{\bfseries --}\, Low interpretability\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Sample-inefficient\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Sensitive to search} \\
\addlinespace[2pt]


\textbf{\fcirc{Blue}{2}}\;\prog &
\cellcolor{lightBlue!12}\makecell[l]{Text\\critique} &
\cellcolor{lightBlue!8}$\text{Refine}(p_t, c_t)$ &
\cellcolor{lightBlue!6}\makecell[l]{Self-Refine~\citep{madaan2023self}\\
Reflexion~\citep{shinn2023reflexionlanguageagentsverbal}\\
Critic~\citep{gou2024critic}\\
ACE~\citep{zhang2025agenticcontextengineeringevolving}} &
\makecell[l]{\posbar \textcolor{green!55!black}{\bfseries +}\, Interpretable edits\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Targeted correction\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Reusable feedback} &
\makecell[l]{\negbar \textcolor{red!70!black}{\bfseries --}\, Critique can be noisy\\
\negbar \textcolor{red!70!black}{\bfseries --}\, May drift\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Validator-dependent} \\
\addlinespace[2pt]


\textbf{\fcirc{Blue}{3}}\;\prog &
\cellcolor{lightBlue!12}\makecell[l]{Selection\\signal} &
\cellcolor{lightBlue!8}$\text{Evolve}(P_t,\text{Fit})$ &
\cellcolor{lightBlue!6}\makecell[l]{Promptbreeder~\citep{fernando2024promptbreeder}\\
STOP~\citep{zelikman2024self}\\
GPTSwarm~\citep{10.5555/3692070.3694667}\\
AutoDAN~\citep{liu2024autodan}\\
Evol-Instruct ~\citep{xu2024wizardlm}\\
GEPA~\citep{agrawal2025gepareflectivepromptevolution}} &
\makecell[l]{\posbar \textcolor{green!55!black}{\bfseries +}\, Strong exploration\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Maintains diversity\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Escapes local optima} &
\makecell[l]{\negbar \textcolor{red!70!black}{\bfseries --}\, Compute-heavy\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Fitness is domain-tuned\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Population drift} \\
\addlinespace[2pt]


\textbf{\fcirc{Blue}{4}}\;\prog &
\cellcolor{lightBlue!12}\makecell[l]{Textual\\gradient} &
\cellcolor{lightBlue!8}$p_t \oplus g(p_t)$ &
\cellcolor{lightBlue!6}\makecell[l]{APO~\citep{pryzant2023automaticpromptoptimizationgradient}\\
TextGrad~\citep{yuksekgonul2024textgradautomaticdifferentiationtext}\\
metaTextGrad~\citep{xu2025metatextgradautomaticallyoptimizinglanguage}\\
SkillOpt~\citep{yang2026skilloptexecutivestrategyselfevolving}} &
\makecell[l]{\posbar \textcolor{green!55!black}{\bfseries +}\, Directional updates\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Often sample-efficient\\
\posbar \textcolor{green!55!black}{\bfseries +}\, Highly automated} &
\makecell[l]{\negbar \textcolor{red!70!black}{\bfseries --}\, Brittle gradients\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Quality varies by LLM\\
\negbar \textcolor{red!70!black}{\bfseries --}\, Limited guarantees} \\

\bottomrule
\end{tabularx}


\renewcommand{\arraystretch}{1.15}

\label{tab:prompt_optimization_paradigms}
```

## Figure 8: fig:agent_memory

Caption: Overview of memory for self-improving agent.

![Overview of memory for self-improving agent.](figures/agent_memory.png)

## Table 2: tab:mem_object_scorecard

Caption: Memory-object scorecard (qualitative). Blue squares indicate an ordinal, literature-grounded assessment (1=low, 5=high) of typical tendencies for each memory object type, synthesized from representative system designs and reported failure analyses rather than from a single standardized benchmark.

```tex
[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt} 
\renewcommand{\arraystretch}{1.25}

\caption{Memory-object scorecard (qualitative). Blue squares indicate an ordinal, literature-grounded assessment (1=low, 5=high) of typical tendencies for each memory object type, synthesized from representative system designs and reported failure analyses rather than from a single standardized benchmark.}
\vspace{0.35em}

\rowcolors{2}{lightBlue!18}{white}


\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}{>{\centering\arraybackslash}m{1.05cm}} 

\begin{tabularx}{\textwidth}{
L{2.05cm}  
L{2.45cm}  
R R R R R  
X         
}
\toprule
\textbf{Object type} &
\makecell[l]{\textbf{Best-for}\\[-1pt]\textbf{persistence}} &
\textbf{Fidelity} &
\makecell[c]{\textbf{Interpre-}\\[-1pt]\textbf{tability}} &
\textbf{Compact} &
\makecell[c]{\textbf{Write}\\[-1pt]\textbf{cost}} &
\makecell[c]{\textbf{Audit-}\\[-1pt]\textbf{tability}} &
\makecell[l]{\textbf{Most common}\\[-1pt]\textbf{failure modes}} \\
\midrule


\makecell[l]{Processed\\trails} &
\makecell[l]{\bbull lessons\\[-1pt]\bbull routines\\[-1pt]\bbull summaries} &
\rate{3}{5} & \rate{5}{5} & \rate{4}{5} & \rate{3}{5} & \rate{5}{5} &
\makecell[l]{\fbull summary bias\\[-1pt]\fbull stale heuristics\\[-1pt]\fbull weak credit assignment} \\

\makecell[l]{Curated raw\\content} &
\makecell[l]{\bbull evidence\\[-1pt]\bbull exact artifacts} &
\rate{5}{5} & \rate{5}{5} & \rate{1}{5} & \rate{4}{5} & \rate{5}{5} &
\makecell[l]{\fbull context bloat\\[-1pt]\fbull retrieval noise\\[-1pt]\fbull privacy leakage surface} \\

\makecell[l]{Integrated external\\knowledge} &
\makecell[l]{\bbull shared factual state\\[-1pt]\bbull grounding} &
\rate{4}{5} & \rate{4}{5} & \rate{3}{5} & \rate{4}{5} & \rate{4}{5} &
\makecell[l]{\fbull grounding failure\\[-1pt]\fbull staleness / inconsistency\\[-1pt]\fbull tool brittleness} \\

\makecell[l]{Latent\\embeddings} &
\makecell[l]{\bbull associative carryover\\[-1pt]\bbull fast recall} &
\rate{3}{5} & \rate{1}{5} & \rate{5}{5} & \rate{2}{5} & \rate{1}{5} &
\makecell[l]{\fbull drift / contamination\\[-1pt]\fbull hard-to-debug retrieval\\[-1pt]\fbull silent corruption} \\

\bottomrule
\end{tabularx}

\label{tab:mem_object_scorecard}
```

## Table 3: tab:memory_architecture

Caption: Memory architecture and processing operators. Checkmarks indicate the memory object and structure choices reported by each system. Dots denote the relative emphasis of a mechanism as primary (), secondary (), or absent/unclear (). Processing is summarized by CRUD (Create, Read, Update, Delete). We further characterize governance along two dimensions: Select, which determines what information is written and retrieved based on saliency and utility, and Maintain, which sustains memory quality over long horizons through consolidation, refresh, and forgetting.

| m3.0cm *12>0.92cm tab-blue 1>3.0cm 2*white -0.15 | 2cwhite Object | 4cwhite Structure | 4cwhite Processing | 2cwhite Governance |
| --- | --- | --- | --- | --- |
| (lr)2-3(lr)4-7(lr)8-11(lr)12-13 | blue!12 |  |  |  |
| Objects | blue!12 |  |  |  |
| Objects | blue!12Flat | blue!12Hier. | blue!12Graph | blue!12 |
| Retr. | blue!12 |  |  |  |
| C-0.35em | blue!12 |  |  |  |
| R | blue!12 |  |  |  |
| U | blue!12 |  |  |  |
| D | blue!12Select | blue!12Maint. |  |  |
| .lanchantin2023learningreasonmemorizeselfnotes[l]Self-Notes |  |  |  |  |
| (2023) | -- | -- | -- | -- |
| .park2023generativeagentsinteractivesimulacra[l]Generative |  |  |  |  |
| Agents (2023) | -- | -- | -- |  |
| .guan2024richelieu[l]Richelieu |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .suzgun2025dynamiccheatsheettesttimelearning[l]DC |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .liang2025selfevolvingagentsreflectivememoryaugmented[l]SAGE |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .chhikara2025mem0buildingproductionreadyai[l]Mem0 |  |  |  |  |
| (2025) | -- | -- | -- |  |
| .salama2025meminsightautonomousmemoryaugmentation[l]MemInsight |  |  |  |  |
| (2025) | -- | -- | -- |  |
| .zhang2025memgenweavinggenerativelatent[l]MemGen |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .zhang2025agenticcontextengineeringevolving[l]ACE |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .xu2025amemagenticmemoryllm[l]A-MEM |  |  |  |  |
| (2025) | -- | -- | -- |  |
| .wang2024agentworkflowmemory[l]AWM |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .ouyang2025reasoningbankscalingagentselfevolving[l]Reasoning |  |  |  |  |
| Bank (2025) | -- | -- | -- | -- |
| .lee2024humaninspiredreadingagentgist[l]ReadAgent |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .long2025seeinglisteningrememberingreasoning[l]M3-Agent |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .zhao2024expel[l]ExpeL |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .tran2025primeplanningretrievalintegratedmemory[l]PRIME |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .zhang-etal-2024-codeagent[l]CodeAgent |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .dillon2025contextualmemoryreweavinglarge[l]CMR |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .wang2024memoryllmselfupdatablelargelanguage[l]MemoryLLM |  |  |  |  |
| (2024) | -- | -- | -- | -- |
| .wang2025mextendingmemoryllmscalable[l]M+ |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .sun2025hierarchicalmemoryhighefficiencylongterm[l]H-MEM |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .koley2025salmmultiagentframeworklanguage[l]SALM |  |  |  |  |
| (2025) | -- | -- | -- | -- |
| .cheng2022xmemlongtermvideoobject[l]XMem |  |  |  |  |
| (2022) | -- | -- | -- | -- |
| .song2024moviechatdensetokensparse[l]MovieChat |  |  |  |  |
| (2024) | -- | -- | -- |  |
| .helmi2025decentralizingaimemoryshimi[l]SHIMI |  |  |  |  |
| (2025) | -- | -- | -- |  |

## Figure 9: fig:full_sacffolding

Caption: Full scaffolding self-improvement across iterations.

![Full scaffolding self-improvement across iterations.](figures/full_sacffolding.png)

## Figure 10: fig:application

Caption: Representative application domains for self-improving agents.

![Representative application domains for self-improving agents.](figures/application.png)

## Table 4: tab:applications_loops

Caption: Application arenas viewed as self-improvement loops. Each domain induces a characteristic sandbox and learning signal, which shapes the dominant bottlenecks, the primary improvement target, and the iteration mode.

```tex
[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.10}
\rowcolors{3}{gray!8}{white}


\begingroup
\renewcommand{\tabularxcolumn}[1]{m{#1}}
\setlength{\emergencystretch}{1em}
\sloppy

\begin{tabularx}{\textwidth}{
>{\centering\arraybackslash}m{1.8cm}  
>{\raggedright\arraybackslash}X       
>{\raggedright\arraybackslash}X       
>{\raggedright\arraybackslash}X       
>{\raggedright\arraybackslash}X       
>{\raggedright\arraybackslash}X       
>{\raggedright\arraybackslash}X       
}
\toprule
\rowcolor{tab-blue}
\textcolor{white}{\textbf{Domain}} &
\textcolor{white}{\textbf{Sandbox and arena}} &
\textcolor{white}{\textbf{Learning signal}} &
\textcolor{white}{\textbf{Main bottleneck}} &
\textcolor{white}{\textbf{Primary improvement target}} &
\textcolor{white}{\textbf{Iteration mode}} &
\textcolor{white}{\textbf{Exemplars}} \\
\midrule

\domSWE &
\cellwrap{
  \bi{Repository with compiler, unit tests, and CI}
  \bi{Failures are typically reversible}
} &
\cellwrap{
  \bi{Deterministic binary outcomes (pass or fail)}
  \bi{Compilation errors}
  \bi{Static analysis signals}
} &
\cellwrap{
  \bi{Patch correctness under repository constraints}
  \bi{Tool and interface efficiency}
} &
\cellwrap{
  \bi{Mainly scaffolding}
  \bi{Some systems also self-edit code or fine-tune models}
} &
\cellwrap{
  \bi{Online debugging per issue}
  \bi{Offline aggregation across issues}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.zhang2025darwingodelmachineopenended}{\makecell[l]{DGM(2025)}}}}
  \li{\textbf{\hyperlink{cite.wang2025huxleygodelmachinehumanlevelcoding}{\makecell[l]{HGM(2025)}}}}
  \li{\textbf{\hyperlink{cite.xia2025livesweagentsoftwareengineeringagents}{\makecell[l]{Live-SWE-agent\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.zhang2026agentdevelreframingselfevolvingllm}{\makecell[l]{AgentDevel\\(2026)}}}}
} \\

\domWeb &
\cellwrap{
  \bi{Simulated and standardized browsers}
  \bi{Partial observability of user interfaces}
} &
\cellwrap{
  \bi{Sparse task completion signals}
  \bi{Long-horizon failures}
  \bi{Partial checks and heuristics}
} &
\cellwrap{
  \bi{Grounding actions to dynamic layouts}
  \bi{Distribution shift over sites and pages}
} &
\cellwrap{
  \bi{Scaffolding, including perception--action grounding, planning, and iterative repair}
  \bi{Trajectory and trace curation}
} &
\cellwrap{
  \bi{Imitation style learning}
  \bi{Online correction along trajectories}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.qi2025webrl}{\makecell[l]{WebRL(2025)}}}}
  \li{\textbf{\hyperlink{cite.fang2025webevolver}{\makecell[l]{WebEvolver\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.zheng2025skillweaverwebagentsselfimprove}{\makecell[l]{SkillWeaver\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.zhang2026webrollbackenhancingwebagents}{\makecell[l]{WebRollback\\(2026)}}}}
} \\

\domGame &
\cellwrap{
  \bi{Game engines with reliable reset}
  \bi{Self-play interactions}
} &
\cellwrap{
  \bi{Win or loss outcomes, or scalar rewards}
  \bi{Clear terminal signals}
} &
\cellwrap{
  \bi{Long-horizon planning}
  \bi{Imperfect information in some settings}
  \bi{Multi-agent non-transitivity}
} &
\cellwrap{
  \bi{Model and policy parameters}
  \bi{Search and planning components}
} &
\cellwrap{
  \bi{Self-play}
  \bi{Iterative policy improvement}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.guan2024richelieu}{\makecell[l]{Richelieu(2024)}}}}
  \li{\textbf{\hyperlink{cite.xu2025dipllmfinetuningllmstrategic}{\makecell[l]{DipLLM(2025)}}}}
  \li{\textbf{\hyperlink{cite.yuan2026mars}{\makecell[l]{MARSHAL\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.cheng2025selfplayingadversariallanguagegame}{\makecell[l]{SPAG(2025)}}}}
} \\

\domSci &
\cellwrap{
  \bi{Tool augmented research loops}
  \bi{Variable-cost evaluation}
} &
\cellwrap{
  \bi{Experimental metrics}
  \bi{Tool outputs}
  \bi{Critique-based refinement}
} &
\cellwrap{
  \bi{Expensive and noisy evaluation}
  \bi{Knowledge fragmentation}
  \bi{Heterogeneous tools and interfaces}
} &
\cellwrap{
  \bi{Scaffolding, including tool orchestration, planning, and verification}
  \bi{Domain specialization}
} &
\cellwrap{
  \bi{Propose, run, critique, and revise cycles}
  \bi{Mixed online and offline iteration}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.lu2024aiscientistfullyautomated}{\makecell[l]{The AI Scientist\\(2024)}}}}
  \li{\textbf{\hyperlink{cite.yamada2025aiscientistv2workshoplevelautomated}{\makecell[l]{AI-Scientist-v2\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.ghafarollahi2024sciagentsautomatingscientificdiscovery}{\makecell[l]{SciAgents(2024)}}}}
  \li{\textbf{\hyperlink{cite.gottweis2025aicoscientist}{\makecell[l]{AI co-scientist\\(2025)}}}}
} \\

\domRobot &
\cellwrap{
  \bi{Simulators with limited real-world rollouts}
  \bi{Safety constraints}
} &
\cellwrap{
  \bi{Rewards and success signals}
  \bi{Real-world data is costly}
} &
\cellwrap{
  \bi{Data collection and safety}
  \bi{Sim-to-real transfer}
  \bi{Dynamics credit assignment}
} &
\cellwrap{
  \bi{Policy and model parameters via a data flywheel}
  \bi{Curricula and safety scaffolds}
} &
\cellwrap{
  \bi{Collect data, retrain, and redeploy}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.bousmalis2023robocatselfimprovinggeneralistagent}{\makecell[l]{RoboCat(2023)}}}}
  \li{\textbf{\hyperlink{cite.zhou2024autonomousimprovementinstructionfollowing}{\makecell[l]{SOAR(2024)}}}}
  \li{\textbf{\hyperlink{cite.xu2024sinvigselfevolvinginteractivevisual}{\makecell[l]{SInViG(2024)}}}}
  \li{\textbf{\hyperlink{cite.yuan2025remacselfreflectiveselfevolvingmultiagent}{\makecell[l]{REMAC(2025)}}}}
  \li{\textbf{\hyperlink{cite.tian2025seear1treestructuredreinforcementfinetuning}{\makecell[l]{SEEA-R1(2025)}}}}  
} \\

\domPC &
\cellwrap{
  \bi{Virtualized desktops}
  \bi{Standardized operating system tasks}
  \bi{Brittle user interfaces}
} &
\cellwrap{
  \bi{Task completion and state checks}
  \bi{Long-horizon objectives}
} &
\cellwrap{
  \bi{Diversity of applications}
  \bi{State tracking}
  \bi{Robust exploration of unseen apps}
} &
\cellwrap{
  \bi{Scaffolding, including hierarchical planning, retrieval, and curricula}
  \bi{Action-trace training}
} &
\cellwrap{
  \bi{Experience reuse and curriculum learning}
  \bi{Iterative trace collection}
} &
\cellwrap{
  \li{\textbf{\hyperlink{cite.wu2024oscopilotgeneralistcomputeragents}{\makecell[l]{OS-Copilot\\(2024)}}}}
  \li{\textbf{\hyperlink{cite.xiao2025uigenie}{\makecell[l]{UI-Genie(2025)}}}}
  \li{\textbf{\hyperlink{cite.wu2025guireflectionempoweringmultimodalgui}{\makecell[l]{GUI-Reflection\\(2025)}}}}
  \li{\textbf{\hyperlink{cite.cheng2026evolvingtasksempoweringmultimodality}{\makecell[l]{SEA(2026)}}}}
} \\

\bottomrule
\end{tabularx}

\endgroup

\caption{Application arenas viewed as self-improvement loops.
Each domain induces a characteristic sandbox and learning signal, which shapes the dominant bottlenecks, the primary improvement target, and the iteration mode.}
\label{tab:applications_loops}
```

## Figure 11: fig:benchmark

Caption: Paper--benchmark incidence matrix for self-improving agents, covering representative benchmarks and methods. Rows enumerate benchmark suites and are grouped by evaluation interface: 0.6pt green!40!blackwhitescaffolding-level benchmarks (interactive, agent-centric) versus 0.6pt RoyalBlue!80!blackwhiteFM-level benchmarks (static, model-centric). Columns are representative self-improving methods; column colors indicate the improvement mechanism used in our taxonomy: red!70!blackFM improvement and yellow!85!blackprompt/ green!60!blackmemory/ RoyalBluetool/ orange!90!blackfull scaffolding self-improvement. A filled cell indicates that the corresponding paper uses the benchmark; ``*'' denotes a benchmark family.

![Paper--benchmark incidence matrix for self-improving agents, covering representative benchmarks and methods. Rows enumerate benchmark suites and are grouped by evaluation interface: 0.6pt green!40!blackwhitescaffolding-level benchmarks (interactive, agent-centric) versus 0.6pt RoyalBlue!80!blackwhiteFM-level benchmarks (static, model-centric). Columns are representative self-improving methods; column colors indicate the improvement mechanism used in our taxonomy: red!70!blackFM improvement and yellow!85!blackprompt/ green!60!blackmemory/ RoyalBluetool/ orange!90!blackfull scaffolding self-improvement. A filled cell indicates that the corresponding paper uses the benchmark; ``*'' denotes a benchmark family.](figures/benchmark_matrix.png)
