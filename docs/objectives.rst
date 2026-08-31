Project Objectives & Scientific Context
========================================

Overview & Mission
------------------

General-purpose Large Language Models (LLMs) frequently suffer from critical hallucinations in scientific, technical, and engineering domains:

1. **Dimensional Inhomogeneity:** Mixing incompatible quantity kinds (e.g., adding energy in Joules to torque in Newton-metres).
2. **Unit Conversion Drift:** Applying incorrect multiplication factors or neglecting affine temperature offsets (:math:`^{\circ}\text{F} \leftrightarrow \text{K}`).
3. **Physical Constant Misquotation:** Truncating or hallucinating empirical physical constants or quoting obsolete pre-2019 defining constant values with non-zero uncertainties.

**DRUM-ML** directly addresses these systemic vulnerabilities by grounding dataset synthesis in formal Semantic Web ontologies and validating all prompt-response pairs through symbolic computation and exact Decimal arithmetic.

.. warning::

   **Experimental Research Prototype — Not for Production Use**

   This project and its outputs are experimental prototypes intended solely for evaluation, scientific research, and educational purposes. They must not be deployed in production or safety-critical applications without accredited verification.

Governance & Leadership
-----------------------

- **Project Lead:** **Pascal Heus** (``pascal@codata.org``)
- **Organization:** `CODATA DRUM Working Group <https://drum.codata.org>`_ (*Digital Representation of Units of Measurement*), under the Committee on Data of the International Science Council (ISC).
- **Contact:** ``drum@codata.org``
- **Primary Deliverables:**
  - **Open-Access Hugging Face Dataset (``drum-ml/metrology-instruct``):** High-quality SFT and DPO splits.
  - **DRUM Metrology Benchmark (M-Eval):** Standardized 6-task evaluation suite with dual MCQ and symbolic tracks, fully integrated with ``lm-evaluation-harness``.
  - **``drum-ml`` Python Package & CLI:** Autonomous CLI for continuous dataset generation, custom ontology synthesis, and benchmark evaluation.

Master Knowledge Repositories
-----------------------------

DRUM-ML ingests from three master authorities:

1. **The SI Reference Point (BIPM SI Digital Framework):**
   - Official digital representation of the International System of Units (9th Edition, 2019 Revision).
   - Ingests SI core ontology, defining constants (:math:`c, h, e, k, N_{\text{A}}, \Delta\nu_{\text{Cs}}, K_{\text{cd}}`), base units, prefixes, and derived units.
   - Endpoint: `https://si-digital-framework.org/SI <https://si-digital-framework.org/SI>`_

2. **CODATA Fundamental Constants Project (DRUM Constants):**
   - Official physical constant values published by NIST and recommended by CODATA (1969–2022).
   - Captures versioned historical changes, standard uncertainties (:math:`u`), relative uncertainties (:math:`u_r`), and exact SI defining flags.
   - Endpoint: `https://api.codata.org/drum/constants <https://api.codata.org/drum/constants>`_

3. **QUDT 2.1 Reference (Quantities, Units, Dimensions, and Types):**
   - Broad semantic graph of units, quantity kinds, 7-base dimension vectors, and conversion formulas.
   - Endpoint: `https://qudt.org/ <https://qudt.org/>`_

Strict Precedence Hierarchy
---------------------------

When entity definitions or conversion factors overlap, DRUM-ML enforces a strict precedence hierarchy:

.. math::

   \textbf{Tier 1 (BIPM SI Framework)} \succ \textbf{Tier 2 (CODATA DRUM Constants)} \succ \textbf{Tier 3 (QUDT 2.1)}
