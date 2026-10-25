# MSc Thesis Defense — Speech Script

This script follows the current 17-slide version of
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
redundant copies. Because routing depends on the input, expert popularity and
rank load can change over time.

## Slide 4 — EPLB

EPLB means Expert Parallel Load Balancing. It dynamically balances expert
workloads across GPUs.

It has two main jobs. Expert load balancing copies overloaded experts. Rank
load balancing migrates experts from overloaded ranks to less busy ranks.

## Slide 5 — Qwen3 MoE Architecture

This figure shows where the experts are located in Qwen3 model.

Each decoder layer contains attention and an MoE block. Inside the MoE block,
the gate selects experts for each token. EPLB does not move the attention
weights. It moves the expert weights in these MoE blocks.

Essentially, a expert is a FFN.

## Slide 6 — Sync and Async EPLB

In synchronous EPLB, inference is paused during migration. In this case, one
large batch can finish migration as fast as possible. Splitting it into several
batches would make the inference pause longer. Therefore, my feature is always
disabled in synchronous mode.

In asynchronous EPLB, inference continues while migration runs in the
background. The migration and inference traffic share the same network.

Several batches make the migration itself slower. However, they reduce the
traffic pressure of inference. Users care about inference latency,
not expert migration time. This is the trade-off studied in this thesis.

## Slide 7 — ILP Formulation

I first use an ILP to describe the scheduling problem. It is an offline
reference and is not executed in the serving path.

The primary objective is to lower the maximum inference time during migration.
T infer is the worst inference time predicted by the model.

On the left are all variables and inputs. X p b tells whether flow p is placed
in batch b. Y b tells whether batch b is active. W p is the number of experts
inside flow p. F p e is the fraction of this flow routed through link e. C e is
the link capacity. T zero is the inference time without migration. Rho is the
inference traffic ratio.

Now I will explain the five constraints.

First, every flow runs exactly once.

Second, a rank communicates with at most one peer in one batch. Therefore,
flows in the same batch have disjoint endpoints.

Third, a flow can only use an active batch. Active batches must also be
continuous. For example, we cannot use batch zero and batch two while leaving
batch one empty.

Fourth, L e b is the total migration load routed through link e in batch b.

Fifth, the model uses this link load to set a lower bound on inference time.
T zero is the baseline. C e times one minus rho is the link capacity remaining
for migration. Higher migration load therefore produces a higher predicted
inference time.

The main goal is lower inference time, not minimum migration time.

## Slide 8 — ILP Hotspot Example

This example makes the model easier to understand.

The left graph shows eight GPU ranks on four servers.

The middle graph shows the unbatched case. All flows are submitted together.the inference time is 4.75.

After ILP, the result is four batches.The inference time falls to 2.25.


## Slide 9 — Online Design

The ILP explains the problem and gives an exact result for small examples.
However, ILP is expensive. So we will not use it in serving.

The production design keeps the most useful rule: flows in one batch should
not share a rank.

## Slide 10 — Greedy Algorithm

自由发挥

## Slide 11 — Experimental Setup

I do this experiment on four nodes. Each node had one RTX 6000 GPU with 24
GB memory. The nodes used 10-gigabit Ethernet, without RDMA.

The model was Qwen3-30B-A3B-Instruct-2507 in FP16. Tensor parallelism and
expert parallelism were both four.

I used async EPLB with prefix caching enabled. The benchmark contained
500 prompts at concurrency 32. Each request had a 300-token prefix, 200 more
input tokens, and 300 output tokens.

The off and on cases used the same request stream. 

## Slide 12 — E2E Results

自由发挥

## Slide 13 — Serving Improvements

This chart presents the same result visually.


## Slide 14 — NIC Traffic

I also measured network traffic.
During the benchmark, we record the NIC traffic per second.
So my feature can smooth the large traffic bursts. 

## Slides 15–16 — Profiling
自由发挥
