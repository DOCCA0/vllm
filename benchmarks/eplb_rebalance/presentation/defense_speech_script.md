# MSc Thesis Defense — Speech Script

This script follows the current 16-slide version of
`eplb_migration_batching.pptx`. The target speaking time is about 15 minutes.
Text in square brackets is a presentation action and should not be spoken.

## Slide 1 — Title

Good morning, everyone.

My name is Wu Cao. Today I will present my thesis, *Contention-Aware Expert
Migration for MoE Serving in vllm*.

This work improves expert migration in vLLM. I am proud that it was merged
into the official vLLM project. This means the work is
useful in a real serving system. It is not only an exercise for my thesis.

I would also like to thank my advisor, Professor Massimo Tornatore, and my
co-advisor, Assistant Professor Qiaolun Zhang, for their support.

The main idea is simple. Conflicting transfers are separated. Independent
transfers can still run together.

## Slide 2 — Outline

My talk has four parts.

First, I will introduce MoE, EP, and EPLB. Then I will explain
the contention model and the offline ILP. Next, I will present the online
greedy scheduler. Finally, I will show the serving, network, and profiling
results.

## Slide 3 — MoE and EP

A Mixture-of-Experts model contains many expert networks. For each token, a
router selects only a small subset of them. This is called top-k routing.

With EP, experts are distributed across GPU ranks, and some experts may have
redundant copies. EPLB has two tasks:
- **Expert load balancing:** copy overloaded experts.
- **Rank load balancing:** migrate experts from overloaded ranks to less busy
  ranks.

Because routing depends on the input, both expert popularity and rank load
change over time. EPLB performs these two operations to keep the workload
balanced.

## Slide 4 — Qwen3 MoE Architecture

This figure shows where the experts are located in Qwen3 model.

Each decoder layer contains attention and an MoE block. Inside the MoE block,
the gate selects experts for each token. EPLB does not move the attention
weights. It moves the expert weights in these MoE blocks.

Essentially, a expert is a FFN.

## Slide 5 — Sync and Async EPLB

In synchronous EPLB, inference is paused during migration. In this case, one
large batch can finish migration as fast as possible. Splitting it into several
batches would make the inference pause longer. Therefore, my feature is always
disabled in synchronous mode.

In asynchronous EPLB, inference continues while migration runs in the
background. The migration and inference traffic share the same network.

Several batches make the migration itself slower. However, they reduce the
traffic pressure of inference. Users care about inference latency,
not expert migration time. This is the trade-off studied in this thesis.

## Slide 6 — ILP Formulation

I first use an ILP to describe the scheduling problem. It is an offline
reference and is not executed in the serving path.

The primary objective is to lower the maximum inference time during migration.
Here, T zero is the inference time without migration. C e is the capacity of
link e. L e b is the migration load on link e in batch b. Rho is the inference traffic ratio. So, C e times one minus rho is the capacity still available for migration.


Now I will explain the three constraints.
First, every flow runs exactly once.
Second, a rank communicates with at most one peer in one batch. Therefore,
flows in the same batch have disjoint endpoints.
Third, the migration load L e b is the sum of the flow sizes routed through
link e.

The main goal is lower inference time, not minimum migration time.

## Slide 7 — ILP Hotspot Example

This example makes the model easier to understand.

The left graph shows eight GPU ranks on four servers.

The middle graph shows the unbatched case. All flows are submitted together.the inference time is 4.75.

After ILP, the result is four batches.The inference time falls to 2.25.


## Slide 8 — Online Design

The ILP explains the problem and gives an exact result for small examples.
However, ILP is expensive. So we will not use it in serving.

The production design keeps the most useful rule: flows in one batch should
not share a rank.

## Slide 9 — Greedy Algorithm

自由发挥

## Slide 10 — Experimental Setup

I do this experiment on four nodes. Each node had one RTX 6000 GPU with 24
GB memory. The nodes used 10-gigabit Ethernet, without RDMA.

The model was Qwen3-30B-A3B-Instruct-2507 in FP16. Tensor parallelism and
expert parallelism were both four.

I used async EPLB with prefix caching enabled. The benchmark contained
500 prompts at concurrency 32. Each request had a 300-token prefix, 200 more
input tokens, and 300 output tokens.

The off and on cases used the same request stream. 

## Slide 11 — E2E Results

自由发挥

## Slide 12 — Serving Improvements

This chart presents the same result visually.


## Slide 13 — NIC Traffic

I also measured network traffic.
During the benchmark, we record the NIC traffic per second.
So my feature can smooth the large traffic bursts. 

## Slide 14 — Profiling Method
自由发挥

## Slide 15 — Profiling Results

Each rank recorded 34 scheduler calls. The median cost is about seven to eight
milliseconds per call. Remember that one call covers all 48 layers.

The largest cumulative cost is 302.081 milliseconds on rank zero. In the clean
E2E experiment, batching saved 105.932 seconds.

So the conservative scheduler-cost-to-serving-saving ratio is only 0.285
percent. The extra CPU work is very small compared with the measured serving
benefit.

This is not a full causal breakdown. However, it directly answers whether the
greedy scheduling overhead is worth paying in this experiment.

## Slide 16 — Conclusion and Questions

To conclude, asynchronous expert migration should be scheduled together with
serving. It should not be treated only as a copy task that must finish as fast
as possible.

This work uses an offline ILP to explain the contention objective. It then
turns the main constraint into a lightweight, rank-disjoint greedy scheduler
for vLLM.

In the experiment, throughput improved by 2.28 percent. P50 end-to-end latency
improved by 3.18 percent. NIC P99 traffic fell by 10.24 percent. The measured
scheduler cost was only 0.285 percent of the serving time saved.

The implementation is opt-in for asynchronous EPLB and disabled for
synchronous EPLB. It was merged into upstream vLLM.

Thank you for listening. I am happy to answer your questions.
