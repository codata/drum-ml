Methodology & Metrological Foundations
======================================

The Quantity vs. Unit Dichotomy
-------------------------------

In formal metrology (VIM3 and ISO 80000), **Quantities** (and **Quantity Kinds**) and **Units** are distinct ontological categories:

- **Quantity / Quantity Kind:** An abstract physical property of a phenomenon, body, or substance (e.g., *Energy*, *Torque*, *Frequency*, *Absorbed Dose*). Governed by dimensional equations :math:`[Q] = \text{L}^a \text{M}^b \text{T}^c \dots`
- **Unit:** A real scalar quantity adopted by convention to express values of quantities of the same kind (e.g., *joule*, *newton-metre*, *hertz*, *gray*). Governed by scale multipliers and offsets.

Why This Separation is Fundamental:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multiple distinct quantity kinds share identical dimension vectors:

1. **Torque vs. Energy / Work:** Both share :math:`[\text{L}^2 \text{M} \text{T}^{-2}]`. Expressing torque in **joules** (:math:`\text{J}`) is a severe metrological violation. Torque must be in **newton metres** (:math:`\text{N}\cdot\text{m}`).
2. **Frequency vs. Activity vs. Angular Velocity:** All share :math:`[\text{T}^{-1}]`. Periodic frequency is in **hertz** (:math:`\text{Hz}`), radioactive decay in **becquerels** (:math:`\text{Bq}`), and angular speed in **radians per second** (:math:`\text{rad/s}`).
3. **Absorbed Dose vs. Dose Equivalent:** Both share :math:`[\text{L}^2 \text{T}^{-2}]`. Absorbed radiation dose is in **grays** (:math:`\text{Gy}`), biological damage risk in **sieverts** (:math:`\text{Sv}`).

DRUM-ML enforces this distinction using the ``QuantityKindEntity`` discriminator across all archetypes and validation tiers.

The 6 Pedagogical Archetypes
----------------------------

1. **Direct Identification & Symbol Mapping:** Official SI/QUDT symbols, quantity kinds, defining constants.
2. **Dimensional Decomposition & Base SI Analysis:** Step-by-step physical derivations expanding derived units into base SI dimensions :math:`[\text{L}]^a [\text{M}]^b [\text{T}]^c [\text{I}]^d [\Theta]^e [\text{N}]^f [\text{J}]^g`.
3. **Conversion Chains, Scaling & Temperature Offsets:** Multi-step conversions, prefix composition, and affine temperature offsets (:math:`^{\circ}\text{F} \leftrightarrow ^{\circ}\text{C} \leftrightarrow \text{K}`).
4. **Dimensional Error Detection:** Flagging dimensionally inhomogeneous equations (:math:`x = v_0 t + \frac{1}{2} a t^3`), invalid additions (:math:`10\text{ J} + 5\text{ N}`), and case-sensitivity traps (:math:`\text{mN}` vs :math:`\text{MN}`).
5. **Metrological Tool Use & Serialization:** Translating natural language into SPARQL queries, QUDT models, or executable Python unit code.
6. **Metrological Uncertainty & GUM Propagation:** Calculating combined standard uncertainty :math:`u_c(y) = \sqrt{\sum (\frac{\partial f}{\partial x_i})^2 u^2(x_i)}` and distinguishing exact SI constants (:math:`u=0`).

The 5 Linguistic Personas
-------------------------

1. **Academic Metrologist:** Formal terminology citing BIPM 9th Edition SI brochure and VIM3.
2. **Firmware / IoT Systems Engineer:** Sensor telemetry, ADC bit counts, register scaling, and UCUM format strings.
3. **Data Scientist / ML Analyst:** Dataframe normalization, unit validation in PyTorch/Pandas.
4. **Physics / Engineering Student:** Conceptual confusion, homework questions.
5. **ISO Compliance Auditor:** ISO 17025 / ISO 80000 traceability, calibration certificates, uncertainty budgets.

The 4-Tier Automated Validation Gate
------------------------------------

Every candidate prompt-response pair must pass 4 deterministic audit tiers:

- **Tier 1 (Syntax & LaTeX Compliance):** Balanced delimiters (``$``, ``$$``), valid LaTeX macros, clean JSON escaping.
- **Tier 2 (Symbolic Dimensional Gate):** SymPy and Pint algebraic equivalence and quantity-kind compatibility.
- **Tier 3 (Arbitrary-Precision Decimal Guard):** Python ``decimal.Decimal`` verification of exact 2019 SI defining constants with zero drift tolerance.
- **Tier 4 (Code Sandbox Gate):** Parses emitted SPARQL queries with ``rdflib`` and Python code with ``ast.parse``.

High-Fidelity DPO Negative Mining
---------------------------------

Only candidates where the prompt is coherent and well-formed, but the generated response contains a plausible metrological flaw (e.g. inverted multiplier, case trap, or dropped exponent) are harvested as ``(prompt, chosen, rejected)`` pairs for Direct Preference Optimization (DPO). Unparseable or garbled outputs are strictly discarded.
