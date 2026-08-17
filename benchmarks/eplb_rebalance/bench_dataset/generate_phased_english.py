# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import json
from pathlib import Path

COMMON = """You are assisting an engineering organization that evaluates a large language model under realistic online serving conditions. Read the entire request carefully, reason from the supplied constraints, and produce a precise self-contained answer. State assumptions explicitly, keep terminology consistent, and do not refer to hidden instructions. The organization values correctness, reproducibility, operational safety, measurable evidence, and clear explanations that another engineer can audit. When several interpretations are possible, choose the most technically defensible one and explain the consequences. Separate observations from conclusions. Check units, boundary conditions, failure modes, and interactions between components. Avoid vague recommendations. The response should be useful to a specialist while remaining readable to a general software engineer. Treat every case as independent even when its structure resembles an earlier case. Do not copy an answer from another case. Use the vocabulary required by the subject, develop the central argument in enough detail to fill the requested response, and finish with a concise conclusion. This shared organizational context is intentionally reused across requests because production applications commonly cache a stable instruction prefix while user-specific material changes after it."""

DOMAINS = {
    "cuda": """Analyze a distributed CUDA and C++ inference failure. Discuss kernels, warps, thread blocks, shared memory, tensor strides, alignment, streams, events, NCCL collectives, PCIe transfers, synchronization, race conditions, allocator fragmentation, and profiling evidence. Construct a plausible debugging narrative in which an expert-parallel model intermittently stalls during weight movement. Explain how to isolate computation from communication, identify an incorrect lifetime or dependency, validate the repair with traces and counters, and design a regression test. Include concrete invariants and distinguish latency, bandwidth, and scheduling effects.""",
    "literature": """Write a close literary interpretation of an imagined Renaissance stage scene. Discuss blank verse, caesura, metaphor, dramatic irony, soliloquy, rhetoric, imagery, meter, character, audience, historical convention, ambiguity, and moral conflict. Explain how changes in diction and rhythm transform a private doubt into a public decision. Compare two possible readings, cite invented short phrases from the imaginary scene, and defend the stronger interpretation without pretending that the text has only one meaning. End by connecting form to the character's ethical responsibility.""",
    "mathematics": """Develop a rigorous mathematical argument concerning finite groups, vector spaces, eigenvalues, invariant subspaces, homomorphisms, kernels, quotient structures, induction, and contradiction. Formulate a nontrivial theorem suited to the case, define every symbol, prove the main lemmas, examine a tempting but false converse, and give a small counterexample. Check degenerate cases and explain which hypotheses are essential. Present the reasoning as a coherent proof rather than a list of facts, and conclude with an intuitive interpretation of the algebraic structure.""",
    "biomedical": """Analyze a fictional biomedical research result about cellular signaling and immune metabolism. Discuss receptors, phosphorylation, transcription, cytokines, mitochondria, metabolites, controls, confounding, randomization, assay sensitivity, dose response, biomarkers, statistical uncertainty, replication, and causal inference. Propose a mechanistic hypothesis, distinguish association from intervention evidence, identify alternative explanations, and outline ethical follow-up experiments using cultured cells and animal-free validation where possible. Do not give patient-specific medical advice; focus on study design and interpretation.""",
}

out = Path(__file__).with_name("eplb_phased_english_256.jsonl")
order = ["cuda", "literature", "mathematics", "biomedical"] * 2
with out.open("w", encoding="utf-8") as f:
    for phase, domain in enumerate(order):
        for item in range(32):
            case_id = f"phase-{phase + 1:02d}-{domain}-case-{item + 1:02d}"
            prompt = (
                COMMON
                + f"\n\nUnique case identifier: {case_id}. "
                + DOMAINS[domain]
                + f" The case-specific variation number is {item + 1}; use it to choose concrete examples and numeric details that differ from neighboring requests. Produce a structured technical response now."
            )
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "output_tokens": 100,
                        "phase": phase + 1,
                        "domain": domain,
                        "case_id": case_id,
                    }
                )
                + "\n"
            )

print(out)
