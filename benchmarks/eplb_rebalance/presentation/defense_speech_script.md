# MSc Thesis Defense — Speech Script

This script follows `eplb_migration_batching.pptx` slide by slide.
The words in square brackets are actions. They do not need to be spoken.

## Slide 1 — Title

Good morning, everyone.

My name is Wu Cao.
Today, I will present my thesis project.
The title is *Contention-Aware Expert Migration for MoE Serving*.

This work improves expert migration in vLLM.
I am very proud that this work was merged into the official vLLM project.
For me, this is important.
It shows that the project is useful in a real system.
It is not only work made to finish a thesis.

I would also like to thank my advisor, Professor Massimo Tornatore, and my
co-advisor, Assistant Professor Qiaolun Zhang, for their support.

The basic idea is shown here.
Instead of sending all expert migrations together, I divide conflicting
transfers into different batches.
Independent transfers can still run together.

## Slide 2 — Outline

My talk has four parts.

First, I will introduce Mixture-of-Experts, expert parallelism, and EPLB.
Second, I will explain the network contention problem and the offline ILP
model.
Third, I will present the online scheduling algorithm and its integration into
vLLM.
Finally, I will show the serving results, NIC measurements, and profiling
results.

Let us start with the background.

## Slide 3 — MoE and EP

A Mixture-of-Experts model contains many expert networks.
For each token, a router selects only a small number of experts.
This is called top-k routing.

Because only a few experts run for each token, the model can have many
parameters without using all of them every time.

With expert parallelism, or EP, expert weights are distributed across GPU
ranks.
The example on the right uses four ranks.
Each rank stores a different group of experts.
Some experts may also have extra copies.

Routing depends on the input.
So expert popularity can change over time.
This may overload some ranks while other ranks are less busy.

## Slide 4 — Qwen3 MoE Architecture

This figure shows where MoE appears in the full model.

On the left, we have the complete Qwen3 decoder layer.
Each layer contains an attention block and an MoE block.

The center shows the attention and MLP structures.
The right side shows the MoE block in more detail.
A gate selects the top experts for each token.
The selected expert outputs are then combined.

EPLB does not move the attention weights.
It moves the expert weights inside these MoE blocks.
Qwen3 has many MoE layers.
So one rebalance cycle can create many expert migrations.

## Slide 5 — Asynchronous EPLB

EPLB means Expert Parallel Load Balancing.
It has four main steps.

First, vLLM collects the expert load.
Second, it computes a new expert placement.
Third, it moves the required expert weights.
Finally, it switches to the new placement at a safe point.

In asynchronous EPLB, inference continues while the weights are moving.
This avoids a full serving pause.
However, inference and migration now share the same network.

[Point to the red box.]

If many transfers start together, they may meet at the same rank or NIC.
This creates a traffic burst.
My goal is to reshape this migration traffic.
The expert placement stays the same.
Only the transfer order changes.

## Slide 6 — Sync and Async EPLB

The correct strategy depends on the EPLB mode.

Let us first look at synchronous EPLB.
Inference is paused during migration.
In this case, putting all migrations in one batch is the fastest choice for
serving time.
It finishes migration as soon as possible.
It also makes the inference pause as short as possible.
For this reason, my batching feature is disabled in synchronous mode.

Asynchronous EPLB is different.
Inference continues during migration.
Splitting migration into several batches increases the total migration time.
But in our contention model, it reduces the interference seen by inference.
So inference becomes faster while migration finishes later.

Users care about request latency and serving speed.
They do not usually care if background migration takes a little longer.
Therefore, this trade-off is worthwhile in asynchronous mode.

## Slide 7 — Problem Statement

Now I can define the scheduling problem.

The inputs are the old placement, the new placement, the network topology,
and the available link capacity.

From the placement difference, I obtain directed migration flows.
Each flow has a source rank and a destination rank.

The decision is simple.
I assign every flow to one migration batch.

The objective is to reduce the worst network pressure seen by inference.

There are two main rules.
Every flow must run exactly once.
Also, one rank can join at most one flow in the same batch.
Independent rank pairs can still communicate in parallel.

## Slide 8 — ILP Formulation

I first studied the problem with an Integer Linear Program.
This ILP is an offline reference.
It is not used in the serving path.

Here, x tells us whether flow p is placed in batch b.
Variable y tells us whether a batch is used.
Q is the migration load on one network link.

I will explain the objective first.
[Point to the green formula.]
The objective is to minimize the worst modeled inference time.
This is the same as reducing the highest normalized link pressure across all
links and batches.
Inference traffic already uses part of each link.
The migration should avoid taking too much of the remaining capacity.

Next are the constraints.
[Point to the first formula.]
Every flow must be assigned to exactly one batch.

[Point to the second formula.]
For each batch, one rank can communicate with at most one peer.
This gives rank-disjoint flows.

After finding the best inference value, I use fewer batches only as a
tie-break.
The main goal is still lower interference, not fewer batches.

## Slide 9 — ILP Hotspot Example

This small example explains the result.

The left figure shows eight GPU ranks on four servers.
The servers share an inter-server link.
This shared link is the hotspot.

The middle figure is the unbatched directed migration graph.
All these transfers are submitted together.
The peak normalized link load is 1.50.
The modeled inference time is 4.75.

The right figure shows the ILP result.
It uses four batches.
Transfers inside one batch use different rank pairs.
The peak link load falls from 1.50 to 0.50.
The modeled inference time falls from 4.75 to 2.25.

Migration itself takes longer because the batches run in sequence.
This is expected.
The purpose is to protect active inference.

## Slide 10 — Online Design

The ILP gives an exact answer for the simplified model.
It is useful for understanding the problem.
It is also useful for checking small examples.

However, I do not run the ILP in production.
An online solver adds dependencies and unstable solve time.
The network model is also only an approximation.

Instead, I keep the most useful rule from the ILP.
Flows in one batch should not share a rank.

The online scheduler uses a greedy first-fit method.
It combines experts with the same source and destination.
Then it groups flows that use different ranks.

The result is deterministic and lightweight.
It is also optional.
Most importantly, it does not change expert placement.
It only changes migration order.

## Slide 11 — Scheduling Pipeline

The scheduling pipeline has five steps.

First, I compare the old and new placement arrays.
Second, I find the experts that must arrive from another rank.
Experts that already exist locally do not need network transfer.

Third, I choose source replicas.
The destinations are balanced across available replicas.

Fourth, I coalesce transfers with the same directed rank pair.
For example, two transfers from rank zero to rank one become one flow.
This avoids making the same contention decision twice.

Finally, I apply greedy first-fit grouping.
This creates rank-disjoint migration batches.

The existing placement and source-selection behavior are preserved.
My code changes only the final transfer schedule.

## Slide 12 — Greedy First-Fit

This slide gives a small first-fit example.

Flow zero uses ranks zero and one.
So it enters batch zero.

Flow one uses ranks two and three.
These ranks are different from zero and one.
So flow one can also enter batch zero.

Flow two uses ranks one and two.
It conflicts with both flows in batch zero.
So it starts batch one.

Flow three uses ranks zero and three.
It does not conflict with flow two.
So it can also enter batch one.

In graph terms, each batch is a matching.
No rank appears twice inside the same batch.
This simple rule limits fan-in and fan-out without stopping safe parallelism.

## Slide 13 — vLLM Integration

The implementation separates scheduling from data transfer.

Once per rebalance cycle, the scheduler receives placement arrays for all 48
MoE layers.
It builds all layer schedules in one call.
This greatly reduces Python call overhead.

The 48 schedules are still independent.
Expert data from different layers is not mixed together.

During execution, vLLM processes the layers one by one.
For each layer, it sets the transfer context, adds sends and receives,
executes the batch, and activates the new placement.

NIXL clears its transfer context after executing a batch.
Therefore, the implementation restores the context for every batch.
This preserves real batch boundaries during communication.

## Slide 14 — Experimental Setup

I evaluated the system on four nodes.
Each node had one RTX 6000 GPU with 24 gigabytes of memory.
The nodes used 10-gigabit Ethernet.
NIXL communicated through UCX over TCP.
RDMA was not available.

The model was Qwen3-30B-A3B-Instruct-2507 in FP16.
Tensor parallelism and expert parallelism were both four.

I used asynchronous EPLB.
The prefix cache was enabled.
The test had 500 prompts and concurrency 32.
Each request had a 300-token prefix, 200 more input tokens, and 300 output
tokens.

The off and on cases used the same request stream.
The batching switch was the only intended difference.
All 500 requests completed in both cases.

## Slide 15 — E2E Results

This table shows the end-to-end serving results.
For every metric, a positive improvement is better.

Output throughput increased by 2.28 percent.

TTFT improved by 1.18 percent at P50 and 1.30 percent at P99.
TTFT means time to first token.

TPOT improved by 1.94 percent at P50 and 0.87 percent at P99.
TPOT means time per output token.

End-to-end latency improved by 3.18 percent at P50 and 1.50 percent at P99.

The key point is consistency.
Throughput, TTFT, TPOT, and end-to-end latency all improved.
There was no regression in the reported asynchronous serving metrics.

## Slide 16 — Serving Improvements

This chart shows the same result in a simpler form.

All bars are above zero.
Throughput improves by 2.28 percent.
The latency improvements range from 0.87 percent to 3.18 percent.

These numbers may look small.
But this is a scheduling change in an already optimized serving system.
It does not change the model or the output.
It only reduces interference from background migration.

The largest gain is the P50 end-to-end latency improvement of 3.18 percent.
This suggests that migration batching gives more network headroom to active
requests.

## Slide 17 — NIC Traffic

To check the mechanism, I also measured NIC traffic.

This plot uses one-second receive-plus-transmit samples from the head node.
The gray plot is batching off.
The blue plot is batching on.

Without batching, the P99 traffic rate is 599.81 megabytes per second.
With batching, it is 538.40 megabytes per second.
This is a reduction of 10.24 percent.

The goal is not to reduce every network sample.
The goal is to reduce the large traffic bursts.
Lower tail traffic leaves more network headroom for inference.
This result supports the reason behind the serving improvements.

## Slide 18 — Profiling Method

Next, I measured the CPU cost of the scheduler.

The clean serving benchmark and the profiling run were separate.
This avoids changing the end-to-end A/B result with profiling overhead.

I used NVTX-only Nsight Systems traces on all four Ray workers.
The model, EPLB settings, and workload lengths were the same as in the serving
test.
The profiling run also used 500 prompts.

I added a temporary NVTX range around the scheduler.
This was needed because asynchronous EPLB runs in a background thread.
The range measures scheduler elapsed time only.
It does not include expert transfer time.

This temporary instrumentation was removed from the final production code.

## Slide 19 — Profiling Results

Each rank recorded 34 scheduler calls.
One call builds schedules for all 48 MoE layers.

The P50 cost is about seven to eight milliseconds per call.
The largest cumulative cost is 302.081 milliseconds on rank zero.

The ranks schedule at the same time.
So I do not add the four rank costs together.
For a conservative comparison, I use the largest cumulative rank cost.

In the clean end-to-end experiment, batching saved 105.932 seconds.
The scheduler cost is only 0.285 percent of this saved time.

This comparison is not a full causal breakdown.
However, it shows that the extra CPU scheduling cost is very small compared
with the measured serving benefit.

## Slide 20 — Conclusion

To conclude, expert migration should be scheduled as part of serving.
It should not be treated as an isolated copy task.

This thesis makes four contributions.

First, the model separates inference interference from migration completion
time.
Second, the algorithm builds rank-disjoint migration batches.
Third, the design integrates this scheduler into asynchronous vLLM EPLB.
Fourth, the experiments show improvements in serving and NIC tail traffic.

Throughput improved by 2.28 percent.
P50 end-to-end latency improved by 3.18 percent.
NIC P99 traffic fell by 10.24 percent.

The study still has limits.
It uses one four-node cluster, one model, one backend, and one A/B run per
condition.
More hardware and workloads should be tested in future work.

Finally, the optimization was merged into upstream vLLM.
This confirms that the design is practical and useful beyond this thesis.

Thank you for listening.
I am happy to answer your questions.
