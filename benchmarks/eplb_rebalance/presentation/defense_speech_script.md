# MSc Thesis Defense — Speech Script

This script follows the current 16-slide version of
`eplb_migration_batching.pptx`. The target speaking time is about 15 minutes.
Text in square brackets is a presentation action and should not be spoken.

## Slide 1 — Title

Good morning, everyone.

My name is Wu Cao. Today I will present my thesis, *Contention-Aware Expert
Migration for MoE Serving*.

This work improves expert migration in vLLM. I am proud that it was merged
into the official vLLM project as pull request 52641. This means the work is
useful in a real serving system. It is not only an exercise for my thesis.

I would also like to thank my advisor, Professor Massimo Tornatore, and my
co-advisor, Assistant Professor Qiaolun Zhang, for their support.

The main idea is simple. Conflicting transfers are separated. Independent
transfers can still run together.

## Slide 2 — Outline

My talk has four parts.

First, I will introduce MoE, expert parallelism, and EPLB. Then I will explain
the contention model and the offline ILP. Next, I will present the online
greedy scheduler. Finally, I will show the serving, network, and profiling
results.

## Slide 3 — MoE and EP

A Mixture-of-Experts model contains many expert networks. For each token, a
router selects only a small subset of them. This is called top-k routing.

With expert parallelism, or EP, the experts are distributed across GPU ranks.
The example on the right uses four ranks. Some experts can also have redundant
copies.

Routing depends on the input. Therefore, expert popularity changes over time.
Some ranks may become overloaded while other ranks are less busy. This is the
reason we need dynamic expert load balancing.

## Slide 4 — Qwen3 MoE Architecture

This figure shows where the experts are located in the full Qwen3 model.

Each decoder layer contains attention and an MoE block. Inside the MoE block,
the gate selects experts for each token. EPLB does not move the attention
weights. It moves the expert weights in these MoE blocks.

Qwen3 contains 48 MoE layers. Therefore, one rebalance cycle can create expert
migrations across many layers.

## Slide 5 — Sync and Async EPLB

EPLB means Expert Parallel Load Balancing. It observes expert load, computes a
new placement, and moves expert weights to the required ranks.

[Point to the synchronous case.]

In synchronous EPLB, inference is paused during migration. In this case, one
large batch finishes migration as fast as possible. Splitting it into several
batches would make the serving pause longer. Therefore, my feature is always
disabled in synchronous mode.

[Point to the asynchronous case.]

In asynchronous EPLB, inference continues while migration runs in the
background. The migration and inference traffic share the same network.

Several batches make the migration itself slower. However, they reduce the
traffic pressure seen by active requests. Users care about inference latency,
not whether a background migration finishes a little earlier. This is the
trade-off studied in this thesis.

## Slide 6 — ILP Formulation

I first use an Integer Linear Program to describe the scheduling objective.
The ILP is an offline reference. It is not executed in the serving path.

Each directed flow represents expert data moving from one rank to another.
The variable x assigns a flow to a batch. The other variables describe active
batches and link load.

I will explain the objective first. [Point to the green formula.]

The primary objective is to minimize the worst modeled inference time. In
practice, this means reducing the highest link pressure across all batches.
Migration should not consume too much of the network capacity needed by
inference.

Now the constraints. [Point to the first formula.]

Every flow must execute exactly once.

[Point to the second formula.]

Within one batch, each rank can communicate with at most one peer. Therefore,
flows in the same batch have disjoint endpoints.

After the best objective value is found, fewer batches are used only as a
tie-break. The main goal is lower contention, not minimum migration time.

## Slide 7 — ILP Hotspot Example

This example makes the model easier to understand.

The left graph shows eight GPU ranks on four servers. Some migration paths
share the same inter-server link. This link becomes the hotspot.

The middle graph shows the unbatched case. All flows are submitted together.
The peak normalized load is 1.50, and the modeled inference time is 4.75.

The right graph shows the ILP result with four batches. The peak load falls to
0.50. The modeled inference time falls to 2.25.

The migration takes longer because the batches run in sequence. This is
expected. The purpose is to protect concurrent inference.

## Slide 8 — Online Design

The ILP explains the problem and gives an exact result for small examples.
However, an online solver would add a dependency and unpredictable solve time.

The production design keeps the most useful rule: flows in one batch should
not share a rank.

The scheduler first combines expert transfers with the same source and
destination. It then applies greedy first-fit grouping. The result is
deterministic, lightweight, and optional.

The expert placement does not change. Only the migration order changes. For
lower Python overhead, schedules for all 48 layers are built in one call per
rebalance cycle.

## Slide 9 — Greedy Algorithm

This example shows the greedy algorithm.

Flow zero uses ranks zero and one, so it enters batch zero. Flow one uses ranks
two and three. It is independent, so it also enters batch zero.

Flow two uses ranks one and two. It conflicts with the first batch, so it
starts batch one. Flow three uses ranks zero and three. It does not conflict
with flow two, so it also enters batch one.

In graph terms, every batch is a matching. No rank appears twice in the same
batch. This limits fan-in and fan-out while keeping safe parallel transfers.

## Slide 10 — Experimental Setup

I evaluated the system on four nodes. Each node had one RTX 6000 GPU with 24
gigabytes of memory. The nodes used 10-gigabit Ethernet. NIXL communicated
through UCX over TCP, without RDMA.

The model was Qwen3-30B-A3B-Instruct-2507 in FP16. Tensor parallelism and
expert parallelism were both four.

I used asynchronous EPLB with prefix caching enabled. The benchmark contained
500 prompts at concurrency 32. Each request had a 300-token prefix, 200 more
input tokens, and 300 output tokens.

The off and on cases used the same request stream. The batching option was the
only intended difference. All 500 requests completed in both cases.

## Slide 11 — E2E Results

This table shows the end-to-end serving results. Positive improvement means
that the batching result is better.

Output throughput increased by 2.28 percent.

Time to first token improved by 1.18 percent at P50 and 1.30 percent at P99.
Time per output token improved by 1.94 percent at P50 and 0.87 percent at P99.
End-to-end latency improved by 3.18 percent at P50 and 1.50 percent at P99.

The important point is consistency. Throughput, TTFT, TPOT, and end-to-end
latency all improved. There is no regression in the reported asynchronous
serving metrics.

## Slide 12 — Serving Improvements

This chart presents the same result visually.

All bars are above zero. The gains range from 0.87 to 3.18 percent. This is a
scheduling-only change. It does not change the model, placement, or generated
output. It only reduces interference from background migration.

## Slide 13 — NIC Traffic

I also measured network traffic to check the mechanism behind the serving
improvement.

The plots show one-second receive-plus-transmit traffic samples. Without
batching, the P99 rate is 599.81 megabytes per second. With batching, it falls
to 538.40 megabytes per second. This is a reduction of 10.24 percent.

The purpose is not to reduce every sample. It is to smooth the large traffic
bursts. The lower tail traffic leaves more network headroom for inference.

## Slide 14 — Profiling Method

The greedy scheduler adds CPU work. Therefore, I also need to answer one more
question: is this extra cost small compared with the serving time it saves?

I measure the two sides separately.

First, I add a temporary NVTX range around the cycle-wide greedy scheduler.
Then I use Nsight Systems to record every scheduler call on all four Ray
workers. In the trace, this range is named *eplb: schedule migration batches*.

One scheduler call builds the batches for all 48 MoE layers. I sum the call
times for each rank. Because the ranks work concurrently, I use the largest
rank total instead of adding all four ranks together.

Next, I measure the benefit with a clean asynchronous E2E A/B test. This test
uses the same serving settings, but profiling is disabled. The saved serving
time is the batching-off duration minus the batching-on duration.

Finally, I compare the measured scheduler cost with this saved time. In this
way, profiling overhead does not affect the serving result.

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
