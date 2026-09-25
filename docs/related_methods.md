# Related AI protein and genome methods

This page is a curated landscape of representative methods related to unZipro, updated through **26 September 2026**. It is intentionally selective rather than exhaustive and includes both general protein/genome methods and plant-focused models.

Publication status is stated explicitly:

- **Peer-reviewed**: version of record in a journal or conference proceedings.
- **Preprint**: public manuscript that has not yet been peer reviewed.
- **Software**: code or an online resource accompanying a method.

## Contents

- [How unZipro fits](#how-unzipro-fits)
- [Structure-conditioned sequence and structure design](#structure-conditioned-sequence-and-structure-design)
- [Experimental-feedback and preference alignment](#experimental-feedback-and-preference-alignment)
- [Property guidance and low-data evolution](#property-guidance-and-low-data-evolution)
- [AI agents for protein engineering](#ai-agents-for-protein-engineering)
- [Interaction and network biology](#interaction-and-network-biology)
- [Genome and regulatory sequence models](#genome-and-regulatory-sequence-models)
- [Plant-focused models](#plant-focused-models)
- [Selection principles](#selection-principles)

## How unZipro fits

unZipro is a lightweight structure-aware framework for ranking potentially beneficial mutations. It combines general inverse-folding constraints with family-specific adaptation and is designed to reduce the number of variants that need experimental screening.

This makes unZipro complementary to several neighboring method classes:

- **Inverse-folding models** generate sequences compatible with a supplied backbone.
- **De novo generative models** create new backbones, sequences, or binders.
- **Experimental-feedback methods** update or align a model using measured fitness or preferences over multiple design-build-test-learn rounds.
- **Property-guided methods** steer a generator with predictors or desired annotations.
- **Scientific agents** coordinate multiple models, databases, simulators, and optimization tools.
- **Genome foundation and regulatory models** learn long-range DNA sequence rules and predict variant or regulatory effects.

unZipro primarily addresses mutation prioritization around existing proteins rather than unconstrained generation or a fully automated repeated wet-lab loop.

## Structure-conditioned sequence and structure design

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| ProteinMPNN | 2022 | Peer-reviewed | Message-passing neural network for designing sequences conditioned on a protein backbone | A foundational inverse-folding baseline; unZipro instead emphasizes beneficial-mutation ranking and family adaptation | [Science](https://doi.org/10.1126/science.add2187) · [Code](https://github.com/dauparas/ProteinMPNN) |
| ESM-IF1 | 2022 | Peer-reviewed | Transformer inverse folding learned from millions of predicted structures | A general structure-to-sequence baseline for zero-shot scoring and sequence recovery | [ICML / PMLR](https://proceedings.mlr.press/v162/hsu22a.html) · [Code](https://github.com/facebookresearch/esm) |
| RFdiffusion | 2023 | Peer-reviewed | Diffusion-based generation of protein backbones under structural and functional constraints | Generates new structures, whereas unZipro prioritizes mutations on existing structures | [Nature](https://doi.org/10.1038/s41586-023-06415-8) · [Code](https://github.com/RosettaCommons/RFdiffusion) |
| Chroma | 2023 | Peer-reviewed | Programmable generative model for protein sequence and structure design | Broad de novo generative design rather than focused variant prioritization | [Nature](https://doi.org/10.1038/s41586-023-06728-8) · [Code](https://github.com/generatebio/chroma) |
| BindCraft | 2025 | Peer-reviewed | One-shot binder design using AlphaFold-based hallucination, sequence design, and filtering | A specialized binder-design pipeline; complementary to unZipro's general mutation-ranking workflow | [Nature](https://doi.org/10.1038/s41586-025-09429-6) · [Code](https://github.com/martinpacesa/BindCraft) |

## Experimental-feedback and preference alignment

| Method | Year | Status | Core idea | Experimental feedback | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- | --- |
| ProteinDPO | 2026 | Peer-reviewed | Direct preference optimization aligns protein generators to experimentally measured fitness preferences | Uses paired experimental preferences or fitness comparisons | Learns an aligned generator; unZipro ranks mutations without requiring preference-training rounds | [Nature Methods](https://doi.org/10.1038/s41592-026-03137-3) · [Code](https://github.com/evo-design/protein-dpo) |
| ORI / RLWF | 2026 | Peer-reviewed | **Ontology Reinforcement Iteration** combines an agent, ontology-conditioned protein generation, unified evaluators, and reinforcement learning from wet-lab feedback | RLWF iteratively incorporates experimental assay results | A full closed-loop functional design system; unZipro can serve as a lightweight prioritizer before assays | [Nature Communications](https://doi.org/10.1038/s41467-026-69855-6) |
| RLXF | 2026 | Peer-reviewed | **Reinforcement Learning from eXperimental Feedback** aligns protein language models directly to experimentally measured functional objectives | Wet-lab measurements provide the reinforcement signal | Requires iterative feedback; unZipro is designed to make useful rankings before such a loop is available | [Nature Communications](https://doi.org/10.1038/s41467-026-77557-2) · [Code](https://github.com/RomeroLab/RLXF) |

ORI and RLXF are related in motivation but are distinct frameworks. ORI uses ontology-conditioned generation and its own experimental-feedback reinforcement component (RLWF), whereas RLXF is a separately developed reinforcement-learning framework for protein engineering.

## Property guidance and low-data evolution

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| EVOLVEpro | 2025 | Peer-reviewed | Combines protein language-model representations, small experimental datasets, and iterative model-guided evolution | Shares the goal of reducing experimental burden, but explicitly iterates on low-N measured data | [Nature Methods](https://doi.org/10.1038/s41592-025-02636-1) · [Code](https://github.com/mat10d/EvolvePro) |
| EvoPlay | 2023 | Peer-reviewed | Uses self-play reinforcement learning, a policy-value network, and Monte Carlo tree search to optimize mutation trajectories | Searches multi-step mutation paths; unZipro focuses on direct structure-aware prioritization | [Nature Machine Intelligence](https://doi.org/10.1038/s42256-023-00691-9) · [Code](https://github.com/melobio/EvoPlay) |
| ProteinGuide | 2026 | Peer-reviewed | Provides predictor-based, training-light property guidance for protein sequence generative models, including ESM3 | Guides de novo or masked generation toward target properties; complementary to mutation ranking | [Nature Biotechnology](https://doi.org/10.1038/s41587-026-03207-z) |

## AI agents for protein engineering

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| ProtAgents | 2024 | Peer-reviewed | Multiple LLM agents coordinate physics-based simulation, machine learning, and protein-design tools | Illustrates an orchestration layer into which a mutation-prioritization model could be integrated | [Digital Discovery](https://doi.org/10.1039/D4DD00013G) |
| AutoProteinEngine (AutoPE) | 2024 | Preprint | LLM-driven multimodal AutoML agent for protein sequence and graph learning workflows | Automates model selection and training rather than introducing a new mutation-scoring objective | [arXiv](https://arxiv.org/abs/2411.04440) · [Code](https://github.com/tsynbio/AutoPE) |
| MAProt | 2026 | Peer-reviewed | Multi-agent protein design with Pareto-based negotiation between structure and protein-language-model objectives | Demonstrates multi-objective coordination using tools such as ProteinMPNN, ESM, and SaProt | [AAAI](https://doi.org/10.1609/aaai.v40i2.37142) |
| Pepti-Agent | 2026 | Preprint | MCP-based agent loop for multi-objective peptide generation, prediction, and single-residue refinement | A peptide-specific agent framework; conceptually relevant to tool-based mutation optimization | [arXiv](https://arxiv.org/abs/2606.15422) |

## Interaction and network biology

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| AlphaFold 3 | 2024 | Peer-reviewed | Unified diffusion architecture for protein–protein, protein–nucleic-acid, protein–ligand, ion, and modified-residue complex prediction | Adds interaction and complex-level structural context that can guide interface mutation design | [Nature](https://doi.org/10.1038/s41586-024-07487-w) · [Code](https://github.com/google-deepmind/alphafold3) |
| Geneformer | 2023 | Peer-reviewed | Transfer learning over tens of millions of single-cell transcriptomes for gene dosage, chromatin, and network-dynamics prediction | Represents network biology at the gene/cell level rather than protein structural fitness | [Nature](https://doi.org/10.1038/s41586-023-06139-9) · [Code](https://github.com/ctheodoris/Geneformer) |
| scGPT | 2024 | Peer-reviewed | Generative pretraining over more than 33 million cells for cell annotation, perturbation response, multi-omics integration, and gene-network inference | Connects foundation-model representations to regulatory and perturbation networks | [Nature Methods](https://doi.org/10.1038/s41592-024-02201-0) · [Code](https://github.com/bowang-lab/scGPT) |
| RegFormer | 2026 | Peer-reviewed | Integrates gene-regulatory-network priors with a Mamba architecture for scalable single-cell representation and GRN reconstruction | A recent example of explicitly embedding regulatory network structure into a foundation model | [Nature Communications](https://doi.org/10.1038/s41467-026-72198-x) |

## Genome and regulatory sequence models

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| Evo | 2025 | Peer-reviewed | Long-context biological sequence model spanning molecular to genome scale | Extends generative modeling from proteins to DNA and whole-genome contexts | [Science](https://doi.org/10.1126/science.ado9336) · [Code](https://github.com/evo-design/evo) |
| Evo 2 | 2026 | Peer-reviewed | Genome modeling and design across all domains of life with very long sequence context | A genome-scale foundation model rather than a structure-conditioned protein mutation model | [Nature](https://doi.org/10.1038/s41586-026-10176-5) · [Code](https://github.com/ArcInstitute/evo2) |
| AlphaGenome | 2026 | Peer-reviewed | Unified prediction of regulatory molecular effects and variant consequences from long DNA sequence | Targets noncoding regulatory interpretation, complementary to protein-level engineering | [Nature](https://doi.org/10.1038/s41586-025-10014-0) · [Code](https://github.com/google-deepmind/alphagenome) |

## Plant-focused models

| Method | Year | Status | Core idea | Relationship to unZipro | Paper / code |
| --- | ---: | --- | --- | --- | --- |
| PEAgent | 2026 | Preprint + software | Predicts cell-type-specific chromatin accessibility from DNA sequence in soybean, maize, and rice; extracts cis-regulatory grammar and variant effects | Extends AI-guided plant engineering to noncoding regulation and cell-type-resolved genomics | [bioRxiv](https://doi.org/10.64898/2026.07.22.740070) · [Code](https://github.com/YAOJ-bioin/peagent-analysis) · [Portal](https://www.peagent.org/) |
| PlantPTM | 2026 | Peer-reviewed + software | Integrates protein language models, evolutionary information, and multi-view features to predict nine plant PTM types | Adds residue-level functional annotation that can help interpret or filter plant protein variants | [Molecular Plant](https://doi.org/10.1016/j.molp.2026.08.002) · [Server](https://ai4bio.online/PlantPTM/home/) |

## Selection principles

Methods were selected because they represent at least one of the following:

1. A widely used foundation for inverse folding or de novo protein design.
2. A 2025–2026 advance in experimental-feedback alignment, reinforcement learning, or property-guided generation.
3. An agent architecture that coordinates multiple protein-engineering models or tools.
4. A genome-scale or regulatory model with clear relevance to sequence design and variant interpretation.
5. A plant-focused method that expands the landscape beyond protein sequence design alone.

The list should be updated as peer-reviewed versions replace preprints and as code or trained weights become publicly available.
