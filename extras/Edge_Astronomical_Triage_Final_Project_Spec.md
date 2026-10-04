# Final Project Specification
## Resource- and Contention-Aware Concurrent Edge Triage for Astronomical Transient Candidates

**Course:** Edge Computing  
**Project type:** Systems / Edge-AI development with an R&D-oriented experimental contribution  
**Status:** Proposal and implementation specification  
**Primary domain:** Edge AI + time-domain astronomy  
**Primary task:** Real–bogus astronomical transient candidate triage under constrained edge resources  

---

# 1. Executive Summary

Modern astronomical surveys produce large streams of candidate transient detections. Many detections are not genuine astrophysical events; they are artifacts caused by image subtraction errors, cosmic rays, bad pixels, trails, saturation, registration issues, or other observational effects. A real-time system therefore needs to distinguish **real** astronomical candidates from **bogus** detections before deciding which data are worth preserving or transmitting.

The proposed project builds an **edge-computing triage system** rather than a new astronomical classifier. The system maintains multiple CNN capacity tiers—**Tiny, Medium, and Large**—and dynamically decides:

1. which model tier should process an incoming candidate,
2. whether another inference may safely be launched concurrently,
3. whether an uncertain candidate should be escalated to a stronger tier,
4. how conservatively the output of each model tier should be trusted, and
5. whether the candidate data should be **discarded, compressed, or transmitted**.

The central systems idea is that unused edge resources should not remain idle while a single inference is running. If CPU/GPU/RAM headroom exists, additional candidates may be processed concurrently. However, concurrency must be controlled because naive simultaneous inference can create contention, latency spikes, memory pressure, or out-of-memory failures.

The proposed system therefore combines:

- resource monitoring,
- queue-aware adaptive model selection,
- contention-aware concurrency control,
- uncertainty-driven escalation,
- tier-dependent decision thresholds,
- a recall-constrained data-fate policy.

The project is **not primarily an accuracy-improvement paper**. Its principal claim is expected to be:

> A resource- and contention-aware adaptive edge triage policy can preserve a predefined level of real-transient recall while reducing average compute cost, end-to-end latency, queue growth, and/or transmitted data compared with static single-model or fixed-concurrency baselines.

---

# 2. Working Titles

## Primary working title

**Resource- and Contention-Aware Concurrent Edge Triage for Astronomical Transient Candidates**

## Alternative research-style titles

- **Contention-Aware Adaptive Multi-Model Scheduling for Recall-Constrained Astronomical Transient Triage at the Edge**
- **Recall-Constrained Concurrent Multi-Tier Inference for Astronomical Transient Triage on Resource-Limited Edge Devices**
- **Adaptive Model-Tier and Concurrency Scheduling for Real-Time Astronomical Transient Triage at the Edge**

The final title should be chosen only after implementation clarifies which component produces the strongest experimental contribution.

---

# 3. Background and Motivation

Time-domain astronomy depends on rapidly identifying changing or newly appearing objects. Difference-imaging pipelines compare new observations with reference images and create candidate detections. These candidate streams are dominated by a mixture of genuine astrophysical events and bogus detections.

Sending every candidate and all associated image products to a central server or cloud is undesirable when:

- communication bandwidth is constrained,
- latency matters,
- the candidate arrival rate is bursty,
- the edge device has limited compute and memory,
- continuous connectivity cannot be assumed.

A conventional solution is to deploy a single classifier on the edge. That approach is simple, but it wastes opportunities in two directions:

- a large model may consume excessive compute on easy candidates,
- a small model may be too unreliable for difficult or ambiguous candidates.

A second inefficiency occurs when only one candidate is processed at a time even though the device may have enough spare resources to process another candidate simultaneously.

This motivates a system that adapts both **model capacity** and **safe concurrency** according to the current operating state.

---

# 4. Problem Statement

Existing astronomical real–bogus classifiers can classify transient candidates with high accuracy, and adaptive/multi-model edge-inference systems can select or schedule DNN workloads according to resource constraints. However, these capabilities are generally studied separately.

The proposed problem is:

> How can an edge device process a stream of astronomical transient candidates using multiple model capacities while dynamically exploiting safe parallelism, preserving a required real-transient recall, and reducing compute, latency, queueing, and communication cost?

The challenge is not simply to maximize throughput. Running too many concurrent inferences can increase contention and make the system slower or unstable. Similarly, aggressively selecting only small models can reduce scientific recall.

The system must therefore balance:

- classification quality,
- transient preservation,
- inference latency,
- queue length,
- CPU/GPU/RAM pressure,
- bandwidth,
- concurrent workload,
- uncertainty,
- model trustworthiness.

---

# 5. Research Gap

The literature already contains the following individual ideas:

- real–bogus classification for astronomical transient detection,
- uncertainty quantification for real–bogus classification,
- CNN deployment on constrained/edge hardware,
- adaptive selection among multiple DNN models,
- multi-tenant DNN inference,
- concurrency and GPU scheduling,
- SLO-aware edge inference.

The proposed contribution is **not** that any one of these components is new.

The research gap is the **joint system-level coupling** of:

1. astronomical real–bogus triage,
2. multiple model capacity tiers,
3. resource-aware model-tier selection,
4. dynamic concurrency admission,
5. contention awareness,
6. uncertainty-based escalation,
7. tier-dependent conservative decision thresholds,
8. recall-constrained discard/compress/transmit decisions.

The strongest novelty is expected to come from showing experimentally that this coupling produces a better **recall-versus-resource-cost operating point** than simpler baselines.

---

# 6. Primary Research Question

> Can a resource- and contention-aware multi-tier edge inference scheduler dynamically select model capacity and safe concurrency for astronomical transient candidates while maintaining a fixed scientific/real-transient recall target and reducing resource cost, queueing delay, and transmitted data relative to static and non-adaptive baselines?

---

# 7. Secondary Research Questions

1. **Model-tier question**  
   Can most easy candidates be processed by cheaper CNN tiers while escalating only ambiguous candidates to stronger models?

2. **Concurrency question**  
   Can dynamic concurrency improve throughput and queue stability compared with purely sequential inference without causing unacceptable contention?

3. **Recall question**  
   Can the system maintain a target real-transient survival/recall level while reducing compute and bandwidth?

4. **Coupling question**  
   Do tier-dependent decision thresholds provide measurable benefit over using one fixed threshold for all model tiers?

5. **Uncertainty question**  
   Does uncertainty estimation identify candidates that are disproportionately likely to be misclassified and therefore deserve escalation?

6. **Burst-load question**  
   Does queue-aware adaptation improve end-to-end latency and throughput during candidate-arrival bursts?

---

# 8. Core Hypotheses

## H1 — Adaptive tiering

An adaptive Tiny/Medium/Large strategy will reduce average inference cost relative to always using the Large model while maintaining approximately the same target recall.

## H2 — Dynamic concurrency

Resource-aware dynamic concurrency will improve throughput and reduce queue growth compared with sequential processing, especially during bursty arrivals.

## H3 — Contention awareness

Unrestricted parallel inference will eventually degrade latency or stability, whereas concurrency admission based on CPU/GPU/RAM/queue state will maintain a better operating region.

## H4 — Tier-aware trust

Tier-dependent thresholds will preserve more real candidates at equal resource cost than using identical thresholds across Tiny, Medium, and Large models.

## H5 — Uncertainty-guided escalation

Ambiguous/high-uncertainty samples will contain a higher fraction of model errors than low-uncertainty samples, making selective escalation more efficient than running the Large model on every candidate.

---

# 9. Scientific Task

The immediate machine-learning task is:

**Binary Real–Bogus Classification**

For each astronomical candidate:

- **Real:** likely genuine astrophysical transient/variable source detection.
- **Bogus:** likely artifact or invalid detection.

Important terminology:

- The project should be careful when using the phrase **scientific importance**.
- Real–bogus probability is not equivalent to astrophysical rarity or scientific value.
- Unless the dataset provides a genuine importance/rarity label, the defensible quantity is:
  - **real-candidate preservation**, or
  - **transient survival/recall**, or
  - a triage utility derived from realness + uncertainty.

The proposal should not claim that the system can identify the most scientifically valuable astrophysical events unless an appropriate label or downstream importance model is added later.

---

# 10. Dataset

## Primary planned dataset

**MeerCRAB / MeerLICHT real–bogus candidate data**

The project specification currently assumes use of the public MeerCRAB-associated data release described in the earlier design notes. Before implementation, the exact current downloadable dataset record must be verified and frozen.

The MeerCRAB paper describes optical transient candidates represented through combinations of:

- New image (`N`)
- Reference image (`R`)
- Difference image (`D`)
- Significance image (`S` / Scorr-type information)

The original study trained several network variants and found strong performance for multi-image input configurations.

## Dataset role in this project

MeerCRAB is attractive because:

- it is based on real observatory candidate data,
- it directly matches the real–bogus task,
- it has an established peer-reviewed baseline,
- its original code and pretrained models are available publicly,
- its small image cutouts are suitable for edge CNN experimentation.

## Required dataset exploration before final implementation

The following must be confirmed experimentally rather than assumed:

- exact downloadable dataset record and version,
- licensing,
- exact train/test files,
- number of candidates,
- class counts,
- class imbalance,
- labeling threshold used (`T9`, `T10`, etc.),
- exact image dimensions,
- exact channel order,
- missing values or malformed samples,
- normalization method,
- metadata fields,
- whether the dataset contains predefined train/test splits,
- whether a validation split must be created,
- whether candidates from the same object can leak across splits,
- storage size per candidate,
- raw/compressed byte cost used in bandwidth experiments.

## Split policy

A safe initial policy is:

- preserve the official held-out test split if supplied,
- create a **stratified validation split only from the training data**,
- never tune decision thresholds or ambiguous-zone bounds on the test set.

If the data contain multiple detections from the same astronomical object, object-level splitting should be investigated to prevent leakage.

---

# 11. Base Papers

The current project should treat four papers as the main conceptual anchors. They do not all need to be reproduced fully.

## Base Paper 1 — Bonnet-Guerrini et al. (2026)

**Title:** *Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with Uncertainty Quantification*  
**Status:** arXiv preprint (2026), submitted to Astronomy & Astrophysics.

### Why it matters

This paper provides a recent real–bogus architecture and uncertainty-quantification framework.

The CNN uses three convolutional blocks with channel width controlled by a single parameter:

**F → 2F → 4F**

Each block contains approximately:

- 3×3 convolution,
- batch normalization,
- ReLU,
- 2×2 max pooling,
- spatial dropout.

The convolutional feature extractor is followed by fully connected layers and a sigmoid binary classifier.

The published architecture uses 30×30×2 input in its own dataset and tunes `F`, dense width, dropout, batch size, and learning rate.

### What this project borrows

- scalable same-family CNN design,
- capacity controlled primarily through `F`,
- dropout/UQ ideas,
- MC-dropout evaluation concepts,
- calibration metrics.

### What this project does **not** initially reproduce

- weakly supervised human-label-free training,
- synthetic injection pipeline,
- asymmetric co-teaching,
- the full dual-network training framework.

Those mechanisms solve a different label-noise problem and would significantly increase scope.

### Code status

No official complete public implementation for the paper was identified in the current review. The network architecture is sufficiently specified in the paper to reproduce independently in PyTorch.

---

## Base Paper 2 — Hosenie et al. (2021), MeerCRAB

**Title:** *MeerCRAB: MeerLICHT Classification of Real and Bogus Transients using Deep Learning*  
**Venue:** Experimental Astronomy, 2021.

### Why it matters

This paper anchors:

- the astronomical task,
- the MeerLICHT candidate representation,
- the real/bogus labels,
- baseline model behavior,
- image-channel combinations,
- the intended astronomical context.

### Useful implementation fact

The paper states that code and pretrained models are available through the authors' GitHub/Zenodo release.

### What this project borrows

- dataset/task definition,
- preprocessing guidance,
- original baseline results,
- channel combinations,
- possibly selected architecture components for baseline reproduction.

---

## Base Paper 3 — Taylor et al. (2018)

**Title:** *Adaptive Deep Learning Model Selection on Embedded Systems*  
**Venue:** ACM LCTES 2018.

### Why it matters

This paper directly supports the concept that an embedded system can select among multiple pretrained DNN models according to accuracy/inference constraints.

It was evaluated on NVIDIA Jetson TX2 and has public code.

### What this project borrows

- adaptive model-selection philosophy,
- model portfolio concept,
- comparison against static single-model inference,
- accuracy/latency trade-off framing.

### What is different in this project

Our scheduler additionally considers:

- current resource state,
- queue state,
- concurrency admission,
- uncertainty,
- tier-dependent trust thresholds,
- recall-constrained data fate.

---

## Base Paper 4 — Lee et al. (2026), Quilt

**Title:** *Peak-memory-aware partitioning and scheduling for multi-tenant DNN model inference*  
**Venue:** Journal of Systems Architecture, Vol. 173, 2026.

### Why it matters

Quilt is a recent peer-reviewed multi-tenant DNN inference scheduling paper. It addresses simultaneous execution of multiple DNNs on one GPU and explicitly studies:

- GPU memory pressure,
- out-of-memory risk,
- scheduling granularity,
- multi-model inference,
- task-level scheduling,
- performance under contention.

It evaluates on systems including Jetson Orin Nano.

### What this project borrows

Not the full Quilt compiler.

Instead, Quilt motivates:

- concurrency is not automatically beneficial,
- memory pressure must be considered,
- new work should be admitted only when resource headroom exists,
- scheduling granularity matters,
- multi-model inference should be profiled under contention rather than only in isolation.

---

# 12. Supporting Literature / Secondary Design Sources

These papers should be cited and selectively mined for mechanisms without reproducing their complete systems.

## Convergo (IEEE EDGE 2025)

**Multi-SLO-Aware Scheduling for Heterogeneous AI Accelerators on Edge Devices**

Useful idea:

- scheduling under multiple simultaneous objectives/constraints,
- accuracy,
- throughput,
- deadlines,
- multi-tenancy.

Relevance to this project:

- inspires formulation of **recall + latency + throughput/resource constraints**.

Do not initially reproduce:

- heterogeneous accelerator management,
- complete Convergo scheduler.

---

## PRAS (2025)

Useful idea:

- request/workload-aware scheduling,
- adaptation after requests arrive,
- burst/queue-aware scheduling,
- dynamic response to online workload.

Relevance:

- supports using queue length and arrival pressure as scheduler inputs.

---

## OctopInf (2025)

Useful idea:

- workload-aware inference serving,
- resource allocation,
- scheduling under edge video workloads,
- practical serving architecture.

Relevance:

- can inform worker pools, request management, batching or resource allocation if needed.

---

## BlastNet / CPU–GPU scheduling work

Useful idea:

- workload placement across CPU and GPU,
- deadline-aware multi-DNN inference.

Relevance:

- optional later extension if real Jetson-class hardware is obtained.

---

## Multi-stream / DRL scheduling work

Useful idea:

- GPU concurrent streams,
- scheduling simultaneous inference pipelines.

Relevance:

- conceptual support for parallel inference.

Not recommended for first implementation:

- DRL scheduler,
- layer-level scheduling,
- excessive CUDA-specific complexity.

---

# 13. Model Family

The recommended initial design is a **single scalable CNN family**, not three unrelated architectures.

## Generic architecture

```text
Input candidate cutout
        |
        v
Conv 3x3, F filters
BatchNorm
ReLU
MaxPool 2x2
Spatial Dropout
        |
        v
Conv 3x3, 2F filters
BatchNorm
ReLU
MaxPool 2x2
Spatial Dropout
        |
        v
Conv 3x3, 4F filters
BatchNorm
ReLU
MaxPool 2x2
Spatial Dropout
        |
        v
Flatten
        |
        v
Fully Connected
        |
        v
Fully Connected
        |
        v
Sigmoid
        |
        v
P(real) or P(bogus)
```

## Initial tier concept

| Tier | Illustrative base width | Purpose |
|---|---:|---|
| Tiny | F = 8 | fast, cheap, high-throughput |
| Medium | F = 16 | balanced |
| Large | F = 32 or 64 | strongest model, escalation target |

These values are **not final**.

They must be chosen through:

- parameter count,
- validation performance,
- measured latency,
- measured memory footprint,
- contention behavior.

The Large model does not necessarily need to be extremely large; the tier spacing only needs to produce meaningful cost/quality differences.

---

# 14. Uncertainty Quantification

## Initial method

**MC Dropout**

For selected ambiguous candidates:

1. keep dropout active during inference,
2. perform multiple stochastic forward passes,
3. calculate mean prediction,
4. calculate prediction dispersion/standard deviation,
5. use uncertainty to determine whether escalation or conservative handling is required.

Illustrative:

```python
preds = [model(x, dropout_active=True) for _ in range(N)]
mean_score = mean(preds)
uncertainty = std(preds)
```

## Important implementation concern

Running ~20 stochastic passes can be expensive.

Therefore:

- MC dropout should not necessarily run for every candidate,
- its number of passes should be profiled,
- 5/10/20 passes can be compared,
- a cheaper uncertainty proxy may be used if MC-dropout overhead destroys the compute advantage.

The project must count UQ overhead honestly in total system cost.

---

# 15. Proposed End-to-End Pipeline

```text
                    +-------------------------------+
                    | Astronomical candidate        |
                    | New / Ref / Diff / ... cutout |
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+
                    | Input / Candidate Queue       |
                    +---------------+---------------+
                                    |
                                    v
          +------------------------------------------------+
          | Context & Resource Monitor                     |
          | CPU | RAM | GPU | Queue | bandwidth | latency |
          +----------------------+-------------------------+
                                 |
                                 v
          +------------------------------------------------+
          | Adaptive Scheduler                             |
          |                                                |
          | - choose Tiny / Medium / Large                |
          | - check available resource headroom           |
          | - admit or delay new inference                |
          | - set safe concurrency                        |
          | - enforce per-tier concurrency caps           |
          +----------------------+-------------------------+
                                 |
                    +------------+------------+
                    |            |            |
                    v            v            v
               +---------+  +----------+  +---------+
               | Tiny    |  | Medium   |  | Large   |
               | Frame A |  | Frame B  |  | Frame C |
               +----+----+  +----+-----+  +----+----+
                    |            |             |
                    +------------+-------------+
                                 |
                      Concurrent inference
                                 |
                                 v
          +------------------------------------------------+
          | Prediction / confidence                        |
          +----------------------+-------------------------+
                                 |
                                 v
                    +------------------------+
                    | Ambiguous candidate?   |
                    +------------+-----------+
                                 |
                        +--------+--------+
                        |                 |
                       No                Yes
                        |                 |
                        |                 v
                        |       +----------------------+
                        |       | UQ / MC Dropout      |
                        |       +----------+-----------+
                        |                  |
                        |                  v
                        |       +----------------------+
                        |       | Escalation           |
                        |       | Tiny -> Medium       |
                        |       | Medium -> Large      |
                        |       | Large -> Cloud/Flag  |
                        |       +----------+-----------+
                        |                  |
                        +------------------+
                                 |
                                 v
          +------------------------------------------------+
          | Tier-aware decision thresholds                 |
          | Tiny = most conservative discard threshold     |
          | Medium = moderate                              |
          | Large = more trusted                           |
          +----------------------+-------------------------+
                                 |
                                 v
                    +---------------------------+
                    | Final triage decision     |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
               +---------+   +----------+  +----------+
               | Discard |   | Compress |  | Transmit |
               +---------+   +-----+----+  +-----+----+
                                    |             |
                                    +------+------+
                                           |
                                           v
                                        Cloud

              Feedback:
              active inference changes CPU/GPU/RAM/queue state
                    -> monitor
                    -> scheduler adapts
```

---

# 16. Context / Resource Monitor

The monitor continuously produces an operating-state snapshot.

## Candidate signals

| Signal | Purpose |
|---|---|
| CPU utilization | compute pressure |
| Available RAM | memory safety |
| GPU utilization | accelerator pressure |
| GPU memory | OOM prevention / concurrency safety |
| Input queue length | workload pressure |
| Active Tiny jobs | per-tier concurrency |
| Active Medium jobs | per-tier concurrency |
| Active Large jobs | per-tier concurrency |
| Recent inference latency | detect contention |
| Candidate arrival rate | burst awareness |
| Network bandwidth | transmission planning |
| Network latency | cloud/escalation cost |
| Optional temperature | detect throttling |
| Optional power | energy evaluation |

On PC/simulation:

- `psutil`
- NVIDIA monitoring if GPU is used
- Linux process counters
- custom queue counters
- `tc/netem`
- stress tools

On Jetson later:

- `tegrastats`
- `jtop`
- CUDA/device memory measurements.

---

# 17. Adaptive Scheduler

The scheduler is the main edge systems module.

It makes two major decisions:

## Decision A — Model tier

Choose:

- Tiny
- Medium
- Large

based on current resource and queue state.

## Decision B — Concurrency admission

Choose whether another inference may begin while other inferences are still running.

This is stronger than a fixed worker count.

The scheduler effectively controls:

**C(t) = number of admitted concurrent inference jobs at time t**

subject to resource constraints.

## Initial rule-based version

The first implementation should be deterministic and explainable.

Illustrative only:

```python
if memory_pressure_high or gpu_pressure_high:
    reduce_concurrency()

if queue_high and resource_headroom_exists:
    increase_concurrency()

if queue_low and resources_free:
    allow_stronger_model()

if queue_high:
    prefer_tiny_or_medium()

if active_large >= MAX_LARGE:
    do_not_launch_another_large()
```

Do not treat these thresholds as final.

They must be profiled and calibrated.

---

# 18. Why Parallel / Concurrent Inference Is a Research-Relevant Module

A naive system may process:

```text
Frame 1 -> inference -> finish
Frame 2 -> inference -> finish
Frame 3 -> inference -> finish
```

Even when hardware is partially idle.

The proposed system allows:

```text
Large(Frame 1)
Tiny(Frame 2)
Tiny(Frame 3)
```

when resource headroom exists.

However, maximum concurrency is not automatically optimal.

Possible behavior:

```text
1 worker -> 40 s / 100 candidates
2 workers -> 26 s
4 workers -> 20 s
8 workers -> 29 s
```

The degradation at excessive concurrency may be caused by:

- memory bandwidth contention,
- GPU kernel contention,
- cache contention,
- context switching,
- memory pressure,
- scheduling overhead,
- thermal throttling.

Therefore, the goal is:

> expose useful parallelism without entering the contention-dominated region.

This is the reason dynamic concurrency is more meaningful than simply "using threads."

---

# 19. Queue and Burst Awareness

Astronomical data should be treated as a stream.

The workload generator should support:

- low steady arrival,
- moderate steady arrival,
- high steady arrival,
- bursty arrival,
- alternating low/high periods.

Example:

```text
normal: 2 candidates/s
burst:  15 candidates/s
```

If the queue grows:

- prefer cheaper tiers,
- increase concurrency only if safe,
- avoid launching memory-heavy jobs when headroom is insufficient.

If the queue is nearly empty:

- allow stronger models,
- reduce unnecessary parallelism.

This creates a self-correcting feedback loop.

---

# 20. Tier-Aware Decision Thresholds

A central novelty mechanism is that the same confidence value should not necessarily be trusted equally across model tiers.

Illustrative example:

```python
thresholds = {
    "Large":  {"discard": 0.90},
    "Medium": {"discard": 0.94},
    "Tiny":   {"discard": 0.97},
}
```

The weaker model should require stronger evidence before discarding a candidate.

Exact values must be calibrated on validation data.

This mechanism must be tested through an ablation:

- adaptive scheduler with one common threshold,
- adaptive scheduler with tier-dependent thresholds.

---

# 21. Ambiguous-Zone Escalation

Candidates near the decision boundary should not be handled exactly like obvious examples.

Illustrative logic:

```text
confident -> trust current tier
ambiguous -> UQ / stronger tier
high uncertainty -> escalate
```

Possible escalation:

```text
Tiny -> Medium
Medium -> Large
Large -> flag/transmit/cloud
```

The ambiguous zone must be learned/tuned on validation data.

It is not fixed permanently at values such as 0.70–0.95.

---

# 22. Data-Fate Policy

The final output is not only a class.

The edge system decides:

## Discard

Candidate is judged strongly bogus with sufficiently high trust.

## Compress

Candidate is not important enough for immediate full transmission but should not be completely lost.

Possible future compression options:

- lower-quality image compression,
- transmit only selected channels,
- transmit only metadata + thumbnail,
- buffer compressed data locally.

## Transmit

Candidate should be sent at high priority because it is likely real, uncertain, or otherwise risky to discard.

The initial project may implement **logical compression accounting** before implementing an advanced image codec.

---

# 23. Recall-Constrained Framing

The primary scientific safety constraint is:

> keep at least X% of true real candidates.

Illustrative target:

**real-transient survival recall ≥ 95%**

The exact target must be fixed before final test evaluation.

The system should then minimize cost subject to that target.

This is preferable to simply maximizing the discard rate, because an edge system could trivially reduce bandwidth by incorrectly discarding real candidates.

---

# 24. Hardware and Simulation Strategy

## Current constraint

No physical edge hardware is currently available.

## Important methodological rule

Do **not** claim that a normal PC perfectly emulates Jetson Nano/Orin performance.

An x86 PC with resource limits can simulate:

- constrained CPU,
- constrained RAM,
- limited worker counts,
- queueing,
- bandwidth/latency conditions,
- background contention.

It cannot faithfully reproduce:

- Jetson ARM execution,
- Jetson GPU microarchitecture,
- CUDA scheduling details,
- shared-memory behavior,
- memory bandwidth,
- thermal throttling,
- exact power behavior.

Therefore the correct terminology is:

**software-in-the-loop resource-constrained edge simulation / emulation of operating conditions**

not:

**accurate Jetson hardware emulation**.

## Phase 1 — PC-based controlled environment

Potential tools:

- Docker
- Linux cgroups
- `stress-ng`
- `tc`
- `netem`
- `psutil`
- SimPy
- Python worker pools
- PyTorch

## Phase 2 — Real hardware if available

Preferred:

- Jetson Orin Nano
- Jetson-class university lab hardware

Alternative:

- Raspberry Pi 5 for CPU-only validation

Hardware deployment strengthens the paper but the software-in-the-loop stage should be designed to remain meaningful independently.

---

# 25. Software Stack

## Core

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- psutil

## Concurrency

Initial candidates:

- `asyncio`
- `ThreadPoolExecutor`
- `ProcessPoolExecutor`

GPU implementation later may use:

- CUDA streams,
- PyTorch streams,
- controlled worker processes.

Do not jump directly to CUDA stream programming unless profiling shows it is necessary.

## Simulation / load

- Docker
- cgroups
- stress-ng
- tc/netem
- SimPy

## Experiment tracking

- YAML configuration files
- structured CSV/Parquet logs
- Git commits
- deterministic random seeds.

---

# 26. Logging Schema

Each candidate should produce a structured record containing at least:

```text
candidate_id
arrival_timestamp
queue_enter_timestamp
inference_start_timestamp
inference_end_timestamp
final_decision_timestamp

selected_tier
initial_score
final_score
uncertainty
escalated
escalation_tier

cpu_before
ram_before
gpu_before
gpu_memory_before
queue_length_before
active_jobs_before

concurrency_level
worker_id

decision = discard/compress/transmit
ground_truth

bytes_original
bytes_transmitted

latency_queue
latency_inference
latency_end_to_end
```

This logging is essential because most of the paper's contribution will be proven through system-level measurements.

---

# 27. Experimental Baselines

At minimum, compare:

## B1 — Static Large, Sequential

- always Large,
- concurrency = 1.

Accuracy/reference-quality upper baseline.

## B2 — Static Tiny, Sequential

- always Tiny,
- concurrency = 1.

Cheap lower baseline.

## B3 — Adaptive Model Tier, Sequential

- dynamic Tiny/Medium/Large,
- no parallel inference.

Isolates model-selection gain.

## B4 — Adaptive Tier + Fixed Concurrency

- dynamic model,
- constant worker/concurrency setting.

Isolates whether concurrency itself helps.

## B5 — Adaptive Tier + Dynamic Concurrency

- resource-aware model tier,
- resource-aware concurrency.

Tests scheduler contribution.

## B6 — Proposed Full System

- adaptive tier,
- dynamic concurrency,
- uncertainty escalation,
- tier-dependent thresholds,
- recall-constrained data fate.

---

# 28. Required Ablation Studies

Ablations are critical because the project is primarily a systems integration contribution.

## Ablation A — Tier selection

Adaptive tiering ON vs OFF.

## Ablation B — Concurrency

Sequential vs fixed concurrency vs adaptive concurrency.

## Ablation C — Tier-aware thresholds

Common threshold vs per-tier thresholds.

## Ablation D — Uncertainty

No UQ vs UQ-driven escalation.

## Ablation E — Queue awareness

Resource-only scheduler vs resource + queue scheduler.

## Optional Ablation F — Supporting-literature improvement

For example:

- simple memory headroom rule vs Quilt-inspired memory-aware admission,
- simple objective vs Convergo-inspired multi-constraint priority.

Only implement this if time permits.

---

# 29. Evaluation Metrics

## Classification metrics

Per tier:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient
- False-negative rate
- False-positive rate

Because losing real candidates is important, emphasize:

- recall,
- false-negative rate,
- PR-AUC.

## Calibration / uncertainty

- Brier score
- Expected Calibration Error if implemented
- uncertainty vs error correlation
- error rate among escalated samples
- escalation rate

## System metrics

- candidates/second throughput,
- average queue delay,
- P50/P95/P99 latency,
- end-to-end latency,
- peak queue length,
- queue growth under bursts,
- fraction processed by each tier,
- mean active concurrency,
- peak active concurrency,
- memory usage,
- GPU utilization,
- CPU utilization,
- inference-time degradation under contention,
- OOM/failure count,
- scheduler overhead.

## Resource-efficiency metrics

- average inference cost per candidate,
- total model forward passes,
- FLOPs estimate if useful,
- wall-clock processing time,
- bytes transmitted,
- data reduction rate,
- optional energy per candidate.

## Scientific safety metric

**Real-candidate survival / scientific recall**

This should be a headline constraint.

---

# 30. Headline Plots

## Plot 1 — Recall vs Compute Cost

Compare all major baselines.

Desired result:

high recall, low cost.

## Plot 2 — Throughput vs Concurrency

Demonstrate that throughput initially increases but may decline or saturate after contention.

## Plot 3 — P95 Latency vs Arrival Rate

Show scheduler stability under burst/load conditions.

## Plot 4 — Queue Length Over Time

Compare:

- sequential,
- fixed concurrency,
- adaptive concurrency.

## Plot 5 — Tier Usage Distribution

Show that the scheduler genuinely adapts instead of silently defaulting to one model.

## Plot 6 — Recall vs Transmitted Bytes

Shows whether data triage reduces communication while preserving real candidates.

## Plot 7 — Uncertainty vs Error

Validates whether UQ is useful for escalation.

---

# 31. Fair Comparison Methodology

A key requirement:

**Do not compare system cost at different recall levels and then call the cheapest system better.**

A reckless system can reduce cost simply by discarding more real transients.

Recommended procedure:

1. choose a target recall using domain/project reasoning,
2. calibrate each system on the validation split,
3. tune thresholds so each system approaches the same recall,
4. freeze all thresholds,
5. evaluate once on held-out test data,
6. compare compute, bandwidth, latency, and queue behavior.

If a weak baseline cannot achieve the target recall even when conservative, report its best achievable recall honestly.

---

# 32. Cost Accounting

The full system cost must include:

```text
monitor overhead
+ scheduler overhead
+ initial model inference
+ optional uncertainty passes
+ optional escalation inference
+ compression cost
+ transmission cost if measured
```

Do not report only CNN forward time.

A useful average expression is:

```text
E[cost]
 =
 monitor
 + scheduler
 + E[inference(selected tier)]
 + P(escalation) * escalation_cost
 + P(UQ) * UQ_cost
```

Concurrency complicates cost because wall-clock throughput and per-candidate latency are different quantities. Both should be reported.

---

# 33. Complexity Control / Scope Guardrails

The project should borrow ideas from multiple papers without trying to reproduce multiple full systems.

## Recommended

Use:

- Bonnet-Guerrini → scalable CNN + UQ ideas,
- MeerCRAB → task/data/baselines,
- Taylor et al. → adaptive model selection concept,
- Quilt → memory/contention-aware admission concept,
- PRAS → queue awareness,
- Convergo → multi-constraint/SLO framing,
- OctopInf → practical serving ideas if needed.

## Avoid in first implementation

- full Quilt compiler,
- layer-level model partitioning,
- full Convergo heterogeneous accelerator stack,
- full PRAS bandit logic,
- deep reinforcement-learning scheduler,
- multi-device distributed inference,
- elaborate cloud orchestration,
- advanced learned compression,
- simultaneous implementation of every architecture in the literature.

The main system must remain understandable and ablatable.

---

# 34. Novelty Positioning

The novelty should be stated conservatively.

## Do not claim

- first real–bogus classifier,
- first uncertainty-aware transient classifier,
- first edge deployment of an astronomical classifier,
- first adaptive multi-model system,
- first concurrent DNN inference system.

## Stronger defensible claim

> We integrate multi-tier astronomical real–bogus inference with resource- and contention-aware concurrency admission, uncertainty-based escalation, and tier-dependent recall-constrained data-fate decisions, and evaluate the resulting system under controlled edge resource and streaming workloads.

The final "first" claim must be checked through a dedicated literature search before submission.

---

# 35. Expected Contribution Types

The project may contribute:

1. **System architecture**  
   A complete edge triage pipeline.

2. **Adaptive scheduling policy**  
   Joint tier selection + concurrency admission.

3. **Tier-aware decision coupling**  
   Different discard conservatism according to model tier.

4. **Astronomy-specific safety objective**  
   Real-transient recall constraint.

5. **Experimental evidence**  
   Characterization of the recall/latency/throughput/resource trade-off.

This is a valid systems contribution even if the CNN itself is not novel.

---

# 36. Development Plan

## Stage 1 — Literature and data verification

- download/read base papers,
- verify dataset record,
- inspect original MeerCRAB code,
- inspect available scheduling code if useful,
- produce literature matrix.

## Stage 2 — Dataset exploration

- load data,
- visualize channels,
- measure class distribution,
- verify splits,
- verify preprocessing,
- create validation split.

## Stage 3 — Scalable CNN

- implement F→2F→4F PyTorch family,
- train Tiny/Medium/Large,
- benchmark each model independently.

## Stage 4 — UQ and calibration

- dropout behavior,
- MC-dropout,
- reliability/calibration,
- ambiguous-zone selection.

## Stage 5 — Sequential adaptive system

- resource monitor,
- rule-based tier selection,
- tier thresholds,
- data-fate policy.

## Stage 6 — Concurrency

- input queue,
- worker model,
- simultaneous inference,
- resource contention profiling,
- concurrency caps.

## Stage 7 — Dynamic scheduler

- combine resource state,
- queue state,
- active jobs,
- model tier,
- safe admission.

## Stage 8 — Controlled edge simulation

- CPU/RAM constraints,
- background stress,
- network shaping,
- burst arrivals.

## Stage 9 — Ablations and final experiments

- baselines,
- equal-recall calibration,
- throughput,
- latency,
- queue,
- transmitted bytes.

## Stage 10 — Hardware validation if available

- deploy to Jetson/Raspberry Pi,
- repeat a reduced but representative benchmark.

---

# 37. Early Sanity Checks

Before investing heavily, answer:

1. Is Tiny meaningfully faster than Large?
2. Does Large meaningfully outperform Tiny on the validation set?
3. Can two simultaneous inferences actually improve throughput?
4. At what concurrency does contention begin?
5. Does MC dropout identify difficult samples?
6. Is escalation rare enough that its overhead does not eliminate savings?
7. Does the selected dataset create a meaningful real/bogus imbalance?
8. Are candidate image sizes small enough that scheduling overhead could dominate inference time?

If Tiny and Large have almost identical runtime, or all models are extremely cheap on the development GPU, artificial resource restriction may be required for meaningful edge experiments.

---

# 38. Risks and Mitigations

## Risk: concurrency does not improve throughput

Mitigation:

- treat it as an empirical result,
- profile CPU vs GPU bottlenecks,
- vary workload/model sizes,
- report contention boundaries.

## Risk: MC dropout is too expensive

Mitigation:

- fewer passes,
- use only for ambiguous samples,
- compare cheaper proxies,
- retain it as an ablation rather than mandatory path.

## Risk: all three tiers perform almost identically

Mitigation:

- widen F spacing,
- reduce channels/features for Tiny,
- quantize Tiny later,
- use model distillation only if needed.

## Risk: simulated environment is criticized

Mitigation:

- clearly label as controlled software-in-the-loop edge environment,
- avoid pretending it is Jetson timing,
- obtain limited real-hardware validation if possible.

## Risk: scientific importance claim is unsupported

Mitigation:

- use real-candidate recall/triage utility language,
- do not claim rarity scoring without labels.

## Risk: project becomes too complex

Mitigation:

- rule-based scheduler first,
- one main concurrency architecture,
- secondary papers supply only selected mechanisms.

---

# 39. Minimum Viable Research System

If time becomes limited, the minimum acceptable system is:

- MeerCRAB dataset,
- Tiny/Medium/Large same-family CNNs,
- input queue,
- CPU/RAM/GPU monitoring,
- rule-based model selection,
- dynamic concurrency cap,
- ambiguous-sample escalation,
- tier-dependent thresholds,
- equal-recall comparison,
- controlled load/burst experiments.

Everything else is an extension.

---

# 40. Stretch Extensions

Only after the core system works:

- learned scheduler,
- contextual bandit,
- deadline-aware priority queue,
- candidate urgency score,
- quantized Tiny model,
- TensorRT deployment,
- CUDA streams,
- CPU/GPU placement,
- energy-aware scheduling,
- thermal-aware scheduling,
- genuine scientific-interest classifier,
- cloud verification service.

---

# 41. Proposal Literature-Review Structure

A proposal literature review can be organized as:

## 41.1 Astronomical transient filtering

- difference imaging,
- real–bogus problem,
- MeerCRAB,
- braai,
- ALeRCE,
- related observatory classifiers.

## 41.2 Uncertainty-aware real–bogus classification

- Bonnet-Guerrini 2026,
- MC dropout,
- ensembles/calibration,
- why uncertainty matters before destructive discard.

## 41.3 Adaptive edge inference

- Taylor et al. adaptive model selection,
- static vs dynamic DNN selection,
- accuracy-latency trade-off.

## 41.4 Concurrent / multi-tenant edge inference

- Quilt,
- resource contention,
- memory pressure,
- safe scheduling.

## 41.5 Workload/SLO-aware scheduling

- PRAS,
- Convergo,
- OctopInf,
- queue/load/latency constraints.

## 41.6 Research gap

Conclude that the opportunity is not another standalone classifier or standalone scheduler, but an astronomy-specific, recall-constrained joint triage and concurrency system.

---

# 42. Proposal Introduction Skeleton

A future proposal introduction can follow this flow:

1. Modern transient surveys produce large candidate streams.
2. Difference-imaging pipelines generate many bogus detections.
3. Edge filtering can reduce communication and reaction latency.
4. A single static DNN is inefficient under changing workload/resource conditions.
5. Multiple model tiers allow adaptive quality/cost trade-offs.
6. Concurrent inference can improve utilization but introduces contention.
7. Scientific data cannot be discarded solely for throughput; real-transient recall must be protected.
8. Therefore, this work proposes a resource- and contention-aware adaptive triage system that jointly controls model tier, concurrency, escalation, and data fate.

---

# 43. Expected Final Experimental Claim Template

Do not fill these values until measured:

> At a fixed real-transient survival recall of **X%**, the proposed adaptive tier-and-concurrency scheduler reduced average compute cost by **Y%**, P95 latency by **Z%**, and transmitted data by **W%** relative to the strongest static baseline.

Possible secondary result:

> Dynamic concurrency increased sustainable candidate throughput from **A** to **B candidates/s** while keeping peak resource usage within the configured safety limits.

Possible negative but valid result:

> Beyond concurrency level **C**, contention dominated available parallelism and throughput decreased, motivating dynamic admission rather than fixed maximum parallelism.

---

# 44. Current Open Decisions

The following are intentionally unresolved:

- exact dataset version/record,
- exact MeerCRAB label threshold,
- exact input channel combination,
- Tiny/Medium/Large F values,
- exact number of MC-dropout passes,
- ambiguous-zone bounds,
- target recall value,
- scheduler thresholds,
- concurrency caps,
- CPU vs GPU execution model,
- physical edge hardware availability,
- whether compression is logical accounting or real codec implementation,
- whether scientific-interest scoring is excluded or added later,
- whether Convergo/PRAS/OctopInf mechanisms enter the actual scheduler or remain literature support.

These must be resolved through profiling/experiments, not guessed in the proposal.

---

# 45. References / Starting Reading List

## Core

1. Raphaël Bonnet-Guerrini et al.  
   **Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with Uncertainty Quantification.**  
   arXiv:2607.05393, 2026.  
   https://arxiv.org/abs/2607.05393

2. Zafiirah Hosenie et al.  
   **MeerCRAB: MeerLICHT Classification of Real and Bogus Transients using Deep Learning.**  
   Experimental Astronomy 51, 319–344, 2021.  
   DOI: 10.1007/s10686-021-09757-1  
   arXiv: https://arxiv.org/abs/2104.13950  
   Code: https://github.com/Zafiirah13/meercrab

3. Ben Taylor, Vicent Sanz Marco, Willy Wolff, Yehia Elkhatib, Zheng Wang.  
   **Adaptive Deep Learning Model Selection on Embedded Systems.**  
   ACM LCTES 2018.  
   DOI: 10.1145/3211332.3211336  
   Code: https://github.com/qwerybot/Adaptive_Deep_Learning

4. Jaeho Lee, Ju Min Lee, Haeeun Jeong, Hyunho Kwon, Youngsok Kim, Yongjun Park, Hanjun Kim.  
   **Peak-memory-aware partitioning and scheduling for multi-tenant DNN model inference.**  
   Journal of Systems Architecture, Vol. 173, Article 103696, 2026.  
   DOI: 10.1016/j.sysarc.2026.103696

## Strong supporting paper

5. Ting Jiang et al.  
   **Convergo: Multi-SLO-Aware Scheduling for Heterogeneous AI Accelerators on Edge Devices.**  
   IEEE EDGE 2025.  
   DOI: 10.1109/EDGE67623.2025.00022

## Additional related literature to inspect

- PRAS — adaptive/request-oriented edge inference scheduling (2025)
- OctopInf — workload-aware edge inference serving (2025)
- BlastNet / time-sensitive multi-DNN CPU–GPU edge inference
- braai / ZTF real–bogus classifier
- ALeRCE stamp classifier
- GOTO uncertainty-aware real–bogus work
- Edge SpAIce / static edge deployment comparison

---

# 46. Source-Verification Notes

As of September 2026:

- Bonnet-Guerrini et al. 2026 is publicly available as an arXiv preprint and is described as submitted to Astronomy & Astrophysics.
- The paper explicitly specifies the `F -> 2F -> 4F` architecture.
- A complete official repository for that paper was not identified during the current review; therefore independent PyTorch reproduction is planned if code remains unavailable.
- MeerCRAB is peer-reviewed and has public code/pretrained-model resources.
- Quilt is a peer-reviewed 2026 Journal of Systems Architecture paper and evaluates multi-tenant DNN scheduling including Jetson Orin Nano.
- Convergo is an IEEE EDGE 2025 conference paper and provides useful multi-SLO scheduling framing.

All final proposal citations should be checked again against the actual downloaded papers before submission.

---

# 47. Final One-Paragraph Project Definition

This project proposes an edge-AI system for real-time astronomical transient candidate triage. A stream of real–bogus image cutouts is buffered at the edge and processed using a scalable Tiny/Medium/Large CNN family. A resource monitor observes compute, memory, queue and network state, while an adaptive scheduler jointly chooses model tier and safe concurrent inference level. Easy candidates can be resolved using cheaper tiers; ambiguous or uncertain candidates can be escalated to stronger models. Tier-dependent conservative thresholds protect weaker-model decisions, and a recall-constrained policy determines whether candidate data should be discarded, compressed, or transmitted. The system will be evaluated under controlled resource constraints and bursty workloads using equal-recall baselines, with the primary objective of showing improved resource efficiency, throughput and queue stability without sacrificing real-transient preservation.
