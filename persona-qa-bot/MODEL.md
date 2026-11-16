# A Persona Model for Automated Product Testing

*Research synthesis + proposed model. Compiled 2026-11-16.*

The goal: a persona model whose purpose is **finding defects a real user base
would hit**, not describing a target customer. This doc reviews the relevant
literature, extracts what each line of work contributes, and proposes a
concrete, operational model (a formal schema lives in `persona.schema.json`).

---

## 1. What the research says

### 1.1 Classical personas were built for empathy, not defect-finding
Cooper's personas, formalized by **Pruitt & Grudin (2003)**, are archetypal
users used to *engage a product team* and anchor design decisions. Their own
evidence is about engagement ("personas engage teams more than scenario-based
design"), not bug yield. ([Pruitt & Grudin, *Personas: Practice and Theory*](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/03/pruitt-grudinold.pdf))

**Takeaway for testing:** demographic/empathy personas ("Sarah, 34, busy mom")
are the *wrong* primitive for a test bot. They say nothing about *how* the user
clicks, types, or breaks things. We keep the archetype idea, drop the biography.

### 1.2 Data-driven personas: ground archetypes in real behavior
**Jansen, Salminen et al.** build personas quantitatively from behavioral data
(their APG method uses tens of millions of interactions), via cluster analysis,
PCA, LSA, LDA, or NMF. The pitch is pairing empathy with analytical rigor.
([Data-Driven Personas, Jansen & Salminen](https://www.semanticscholar.org/paper/Data-Driven-Personas-Jansen-Salminen/ec90cede5143c0d7ad6061079a51407c2677b97a);
[Template analysis of 31 quantitative personas](https://link.springer.com/content/pdf/10.1007/978-3-030-50020-7_8.pdf))

**Takeaway:** a testing persona population should be *derived from the product's
own analytics/session logs*, so the mix of personas matches the mix of real
traffic. A bug 60% of real sessions would hit should be found by 60% of agents.

### 1.3 Usage models tell you *where* real users go
Decades of **Markov usage-model / usage-based statistical testing** work builds
a probabilistic model of user sessions from server logs, then generates test
paths weighted by real frequency — explicitly modeling both frequent and rare
behavior, and deriving *worst-case* scenarios for stress/perf testing.
([Usage-based statistical testing](https://www.researchgate.net/publication/220940143_Usage-based_statistical_testing_of_web_applications);
[Worst-case user scenarios via Markov workload models](https://www.researchgate.net/publication/322871273_Identifying_worst-case_user_scenarios_for_performance_testing_of_web_applications_using_Markov-chain_workload_models))
For coverage measurement, **DomCovery** (Mirzaaghaei & Mesbah) measures DOM
**states, transitions, and elements** — a web-native coverage criterion.

**Takeaway:** personas shouldn't wander randomly. Their navigation should follow
a usage model (real transition probabilities), and coverage should be scored in
DOM-state terms, not "did a script pass."

### 1.4 PersonaTester: the first testing-specific persona schema (and the key result)
**PersonaTester / Personified-LLM (FSE 2026)** is the most directly relevant
work. It defines a testing persona along **three behavioral dimensions**, and —
critically — **derives 9 personas from 1,500 annotated real crowdsourced test
traces** rather than inventing them. ([arXiv 2603.24160](https://arxiv.org/abs/2603.24160);
[iGUITest/PersonaTester code](https://github.com/iGUITest/PersonaTester))

| Dimension | Range of values (from the paper) |
|---|---|
| **Testing mindset** | sequential & coherent ⟷ divergent & nonlinear |
| **Exploration strategy** | click-oriented ⟷ core-function-oriented ⟷ input-oriented |
| **Interaction habit** | short valid inputs ⟷ long valid inputs ⟷ invalid inputs |

Result: persona-guided agents produced **more effective test events and found
more crashes (100+) and functional bugs (11)** than a **no-persona baseline** on
15 mobile apps. **This is the core empirical justification for the whole
approach: persona conditioning measurably increases bug yield.**

*Honest caveat (the paper's own threat to validity):* reviewers note it lacks
direct evidence that the agents' traces match *real* crowdworker traces. So
grounding + validation against real sessions matters (see §4).

### 1.5 UXAgent: the architecture to run personas at scale
**UXAgent (CHI 2025)** runs LLM personas as usability-test participants. Its
architecture is the reference design we adopt: ([arXiv 2502.12561](https://arxiv.org/abs/2502.12561))
- **Persona Generator** — produces thousands of varied personas to seed agents.
- **Memory Stream** — stores observations, actions, reflections, and
  spontaneous thoughts, timestamped and scored.
- **Two-loop reasoning (Kahneman-inspired)** — a **Fast Loop** (recency-weighted)
  for real-time action, a **Slow Loop** (relevance-weighted) for deeper
  reasoning/reflection.
- **Universal Browser Connector** — parses raw HTML into a simplified form the
  agent acts on.
- **Result Viewer** — action traces, memory/reasoning traces, and **video**.

The memory scoring comes from **Generative Agents (Park et al., 2023)**:
`score = recency (exponential decay) + importance (LLM 1–10 "poignancy") +
relevance (cosine similarity to query)`, each normalized to [0,1], equally
weighted by default. ([Generative Agents, arXiv 2304.03442](https://arxiv.org/abs/2304.03442))

### 1.6 Big Five as the variation knob — with a reliability warning
Simulation frameworks increasingly parameterize agents by **OCEAN scores (0–1)
rendered into a natural-language prompt** (e.g. AgentWorld; a low-agreeableness,
high-neuroticism user is confrontational — useful for stress tests). Big Five is
preferred over MBTI for reliability/validity. **But** a 2026 psychometric
critique finds LLM "personality" facets collapse (r≥.90) and may not measure a
human-equivalent construct; consistency varies by model.
([LLMs Simulate Big Five, arXiv 2402.01765](https://arxiv.org/abs/2402.01765);
[BIG5-CHAT, arXiv 2410.16491](https://arxiv.org/abs/2410.16491);
[Psychometric critique, arXiv 2607.02325](https://arxiv.org/html/2607.02325))

**Takeaway:** use Big Five to *vary temperament* (patience, caution,
thoroughness), but don't trust it as ground truth — **validate behavior across
repeated runs and against real traces** (§4).

### 1.7 Negative/adversarial testing is a first-class persona class
Classic QA taxonomy covers invalid input, user misuse, environment/system
failure, and adversarial behavior (injection, boundary probing). The persona
angle is strongest in LLM-testing guidance.
([Negative testing](https://ray.run/wiki/negative-testing))

**Takeaway:** a deliberate *minority* of adversarial personas (they don't match
traffic volume, but catch the catastrophic) belongs in the population.

---

## 2. The proposed model: three layers

The synthesis: **ground the population in real data (1.2–1.3), define personas
by testing behavior (1.4) + temperament (1.6), run them with a memory +
dual-loop agent (1.5), and keep an adversarial minority (1.7).**

```
┌─ Layer A · GROUNDING ───────────────────────────────────────────┐
│ Derive the persona POPULATION from the product's own data:      │
│  • analytics funnels + session logs → Markov usage model         │
│  • cluster real sessions (k-means/LDA) → behavioral archetypes   │
│  • each cluster's traffic share → persona "population weight"     │
│  + inject an adversarial minority (not traffic-weighted)         │
└──────────────────────────────────────────────────────────────────┘
                              │  (produces N weighted personas)
┌─ Layer B · PERSONA SCHEMA ──┴────────────────────────────────────┐
│ Each persona = intent + testing-behavior + temperament + context │
│   intent            goal/task they pursue (UXAgent)              │
│   testing_mindset   coherent ⟷ nonlinear        (PersonaTester)  │
│   exploration       click / core-function / input (PersonaTester)│
│   input_habit       short-valid / long-valid / invalid (Persona-)│
│   temperament       OCEAN-derived knobs → patience, caution,     │
│                     error_rate, speed, persistence  (Big Five)   │
│   context           device, network, locale, auth state          │
│   population_weight share of real traffic (priority input)       │
│   grounded_from     the session cluster this was derived from    │
└──────────────────────────────────────────────────────────────────┘
                              │  (each persona drives an agent)
┌─ Layer C · AGENT ───────────┴────────────────────────────────────┐
│  perception → memory → reasoning → action → sensors              │
│   perception   DOM simplified to actionable elements (UXAgent)   │
│   memory       stream: obs/action/reflection/thought,            │
│                score = recency+importance+relevance (Gen.Agents) │
│   reasoning    FAST loop (act now, recency) +                    │
│                SLOW loop (reflect/replan, relevance) (Kahneman)  │
│   action       click/type/scroll/back/submit/wait + explore      │
│   navigation   biased by the Markov usage model, not uniform     │
│   sensors      console errors, failed XHR (frontend→DB sync),    │
│                HTTP 4xx/5xx, layout overflow, DOM-state coverage │
└──────────────────────────────────────────────────────────────────┘
```

### Why this is the right model for *testing* (vs. the alternatives)

- **Over pure random/monkey testing:** usage-model navigation spends effort
  where real users go, so bugs found are bugs that matter (usage-based testing).
- **Over classical personas:** the primitives are behavioral and measurable, so
  they actually change what the agent clicks and types (PersonaTester's result).
- **Over invented LLM personas:** Layer A grounds the population in the product's
  real traffic, addressing PersonaTester's own validity gap and making
  `population_weight` meaningful.
- **Over a flat trait list (what the current skill has):** adds *intent*,
  *memory*, and *usage-weighted navigation* — the difference between a scripted
  click-through and an agent that pursues a goal and adapts.

---

## 3. Priority scoring (grounded)

```
priority(bug) = severity × reach
reach(bug)    = Σ over personas that hit it of  population_weight(persona)
              × hit_frequency across repeated runs
```

Because `population_weight` comes from real traffic (Layer A), reach estimates
**how much of the actual user base hits the bug** — not "how many of my invented
personas did." Adversarial personas carry a small fixed weight so a
security/crash finding still ranks, even at low traffic share.

---

## 4. Reliability & validation (do not skip)

The research is explicit that simulated personality/behavior can be unreliable
(§1.6) and that agent traces may not match real users (§1.4). So the model
*requires*:

1. **Cross-run consistency** — run each persona N times; a finding that appears
   in some-but-not-all runs is **flaky / timing-dependent** (surfaces real race
   conditions, and flags unreliable persona behavior). *(This is already the
   `--repeat` / flaky-detection mechanism in the runner.)*
2. **Trace validation** — where real session logs exist, compare agent
   navigation distributions against real ones (KL divergence over the Markov
   transition matrix). If agents don't move like real users, re-ground Layer A.
3. **Model-sensitivity check** — Big Five expression varies by LLM; pin the
   model and re-measure persona behavior when it changes.

---

## 5. How this maps onto the existing skill, and the upgrade path

| Model element | Status in `persona-qa-bot` today | Upgrade |
|---|---|---|
| Temperament knobs (B) | ✅ `traits` in behaviors.js | map to explicit OCEAN → traits |
| Device / network / context (B) | ✅ device + network profiles | add locale, auth state |
| Sensors + DOM signals (C) | ✅ console/http/request/layout | add DOM-state **coverage** metric |
| Explore mode (C) | ✅ random monkey explore | bias by **Markov usage model** |
| Cross-run / flaky (4) | ✅ `--repeat` + flaky report | add trace-validation (KL) |
| **Intent / goal (B)** | ⚠️ implicit in `goal` string | make it drive an **LLM decision loop** |
| **Memory + dual-loop (C)** | ❌ none (scripted steps) | add memory stream + fast/slow loops |
| **Grounding from real data (A)** | ❌ personas are hand-written | derive population from analytics/logs |
| **Population weight (3)** | ❌ reach = count of personas | weight by real traffic share |
| Adversarial persona class (1.7) | ⚠️ one "security probe" | formal adversarial minority set |

The three ❌ rows are what turn this from a good trait-based harness into the
researched model: **ground the population, give agents intent + memory, and
weight priority by real traffic.**

---

## References

- Pruitt & Grudin (2003), *Personas: Practice and Theory* — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/03/pruitt-grudinold.pdf) · [ACM](https://dl.acm.org/doi/10.1145/997078.997089)
- Jansen & Salminen et al., *Data-Driven Personas* — [Semantic Scholar](https://www.semanticscholar.org/paper/Data-Driven-Personas-Jansen-Salminen/ec90cede5143c0d7ad6061079a51407c2677b97a) · [Template (31 profiles)](https://link.springer.com/content/pdf/10.1007/978-3-030-50020-7_8.pdf)
- Usage-based statistical testing (Markov usage models) — [ResearchGate](https://www.researchgate.net/publication/220940143_Usage-based_statistical_testing_of_web_applications) · worst-case workloads — [ResearchGate](https://www.researchgate.net/publication/322871273_Identifying_worst-case_user_scenarios_for_performance_testing_of_web_applications_using_Markov-chain_workload_models)
- PersonaTester / Personified-LLM (FSE 2026) — [arXiv 2603.24160](https://arxiv.org/abs/2603.24160) · [code](https://github.com/iGUITest/PersonaTester)
- UXAgent (CHI 2025) — [arXiv 2502.12561](https://arxiv.org/abs/2502.12561) · [ACM](https://dl.acm.org/doi/10.1145/3706599.3719729)
- Park et al. (2023), *Generative Agents* — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- Big Five in LLMs — [Simulate Big Five, 2402.01765](https://arxiv.org/abs/2402.01765) · [BIG5-CHAT, 2410.16491](https://arxiv.org/abs/2410.16491) · [psychometric critique, 2607.02325](https://arxiv.org/html/2607.02325)
- Negative testing taxonomy — [QA wiki](https://ray.run/wiki/negative-testing)

*Sourcing note: arxiv.org was not reachable from this environment, so paper
details come from search abstracts, author PDFs on other hosts, and secondary
summaries. Verify exact numbers against the publisher PDFs before citing in
anything formal.*
