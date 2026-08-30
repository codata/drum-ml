.. _scientific_unions_and_personas:

Scientific Unions & Domain Personas
===================================

The **DRUM-ML** pipeline integrates domain-specific persona profiles representing the **CODATA DRUM (`Digital Representation of Units of Measurement <https://drum.codata.org>`_) Working Group** and the **30+ International Scientific Unions & Standards Bodies** of the International Science Council (ISC).

While **Agent 2 (Archetype Scaffolder)** deterministically derives ground-truth answers and dimensional derivations from the BIPM, CODATA, and QUDT master knowledge graphs, **Agent 3 (Linguistic Diversity Augmenter)** projects these scaffolds through **39 specialized scientific personas** to generate authentic, diverse, and contextually rich questions from practitioners in each field.

Overview of Union & Standards Coverage
---------------------------------------

The **39 personas** in DRUM-ML are organized into 5 major disciplinary sectors:

1. Standards Bodies, Metrology Institutes & Global Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Scientific Body / Organization
     - Persona Key (``config.yaml``)
     - Metrological Focus & Domain Coverage
   * - **National Institute of Standards and Technology (NIST)**
     - ``nist_metrologist``
     - NIST SP 330/811 guidelines, Kibble balance realizations, optical clocks, primary calibration standards, CODATA fundamental constants.
   * - **National Research Council Canada (NRC)**
     - ``nrc_metrologist``
     - Quantum Hall resistance (:math:`R_{\text{K}}`), Josephson voltage standards (:math:`K_{\text{J}}`), mass dissemination, precision electrical metrology.
   * - **International Science Council (ISC)**
     - ``isc_science_policy``
     - Interdisciplinary science data interoperability, FAIR digital representation of units, global scientific cooperation across domain unions.
   * - **Academic Metrologist (BIPM / NMI)**
     - ``academic_metrologist``
     - BIPM 9th Edition SI brochure, VIM3 terminology, exact SI 2019 defining constants, formal LaTeX derivations.
   * - **ISO / IEC Calibration Auditor**
     - ``iso_compliance_auditor``
     - ISO/IEC 17025 accreditation, ISO 80000 quantities and units, calibration certificates, GUM uncertainty budgets.
   * - **General User / Everyday Inquirer**
     - ``general_user``
     - Concise, direct questions on common physical units, basic conversions, and fundamental constants.

2. Physical, Chemical & Mathematical Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Scientific Union / Organization
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Pure and Applied Physics (IUPAP)**
     - ``iupap_physicist``
     - IUPAP SUNAMCO symbols, base units, physical dimensions, SI realizations, relativistic kinematics.
   * - **International Union of Pure and Applied Chemistry (IUPAC)**
     - ``iupac_chemist``
     - IUPAC Green Book, amount-of-substance (:math:`\text{mol}, \text{mol/L}`), standard atomic weights, molar gas constant relations.
   * - **International Astronomical Union (IAU)**
     - ``iau_astronomer``
     - Parsecs, light-years, astronomical units (:math:`\text{au}`), solar masses (:math:`M_\odot`), Jansky (:math:`\text{Jy}`) spectral flux densities, stellar photometry.
   * - **International Union of Crystallography (IUCr)**
     - ``iucr_crystallographer``
     - Unit cell dimensions (:math:`\text{Å}, \text{pm}`), reciprocal lattice vectors (:math:`\text{nm}^{-1}`), electron density (:math:`e/\text{Å}^3`), Bragg X-ray angles.
   * - **International Mathematical Union (IMU)**
     - ``imu_mathematician``
     - Dimensional vector spaces, Lie group symmetries in physics, Buckingham :math:`\Pi` theorem proofs.
   * - **Union Radio Scientifique Internationale (URSI)**
     - ``ursi_radio_scientist``
     - RF telemetry, antenna gain (:math:`\text{dBi}`), antenna noise temperature (:math:`\text{K}`), Poynting vectors (:math:`\text{W/m}^2`), plasma frequencies.
   * - **International Union of Theoretical and Applied Mechanics (IUTAM)**
     - ``iutam_mechanics_engineer``
     - Stress tensors (:math:`\text{Pa}`), dynamic/kinematic viscosity (:math:`\text{Pa}\cdot\text{s}, \text{m}^2/\text{s}`), strain energy density, dimensionless hydrodynamic numbers (Reynolds, Mach, Nusselt).
   * - **High-Energy Particle Physics**
     - ``particle_physicist``
     - Electronvolts (:math:`\text{eV, GeV, TeV}`), barn (:math:`\text{b}`) cross-sections, invariant mass (:math:`\text{GeV}/c^2`), natural Planck units.
   * - **Aerospace & Propulsion Dynamics**
     - ``aerospace_propulsion_engineer``
     - Specific impulse (:math:`I_{\text{sp}}` in :math:`\text{s}` or :math:`\text{N}\cdot\text{s/kg}`), thrust (:math:`\text{kN}`), dynamic pressure (:math:`\text{kPa}`), hypersonic flow.
   * - **Firmware & Embedded IoT Engineering**
     - ``firmware_iot_engineer``
     - ADC bit counts, register scaling factors, microcontroller fixed-point units, UCUM format strings, sensor telemetry.
   * - **Data Science & ML Pipeline Engineering**
     - ``data_scientist_analyst``
     - DataFrame unit normalization, PyTorch/Pandas tensor unit verification, multimodal telemetry alignment.
   * - **Physics & STEM Education**
     - ``physics_student``
     - Inquisitive conceptual homework questions clarifying confusing metrological nuances (e.g. torque :math:`\text{N}\cdot\text{m}` vs energy :math:`\text{J}`, frequency :math:`\text{Hz}` vs activity :math:`\text{Bq}`).

3. Earth, Space & Environmental Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Scientific Union / Organization
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Geodesy and Geophysics (IUGG)**
     - ``iugg_geodesist_geophysicist``
     - Gravitational anomalies (:math:`\text{mGal}`), geoid undulations, seismic moments (:math:`\text{N}\cdot\text{m}`), geomagnetic field (:math:`\text{nT}`).
   * - **International Geographical Union (IGU)**
     - ``igu_geographer``
     - Spatial scale, map projections, geospatial attribute scaling, land area units (:math:`\text{ha}, \text{km}^2`).
   * - **International Society for Photogrammetry and Remote Sensing (ISPRS)**
     - ``isprs_photogrammetrist``
     - Ground sampling distance (:math:`\text{cm/px}`), spectral radiance (:math:`\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})`), LiDAR point density (:math:`\text{pts/m}^2`).
   * - **International Society for Digital Earth (ISDE)**
     - ``isde_digital_earth``
     - Discrete Global Grid Systems (DGGS), multi-resolution planetary telemetry harmonization.
   * - **International Union of Soil Science (IUSS)**
     - ``iuss_soil_scientist``
     - Soil bulk density (:math:`\text{g/cm}^3`), cation exchange capacity (:math:`\text{cmol}(+)/\text{kg}`), saturated hydraulic conductivity (:math:`\text{mm/h}`).
   * - **Climate & Energy Systems**
     - ``energy_environmental_scientist``
     - Atmospheric trace gases (:math:`\text{ppm, ppb}`), grid storage (:math:`\text{MWh, GWh}`), solar irradiance (:math:`\text{W/m}^2`), carbon intensity (:math:`\text{g CO}_2/\text{kWh}`).

4. Biological, Medical & Health Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Scientific Union / Organization
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Biological Sciences (IUBS)**
     - ``iubs_biologist``
     - Metabolic scaling rates (:math:`\text{W/kg}`), biomass density, population dynamics, physiological allometry.
   * - **International Union of Immunological Societies (IUIS)**
     - ``iuis_immunologist``
     - International Units (:math:`\text{IU/mL}`) of biological activity, antibody titers, cytokine concentrations, flow cytometry fluorophore intensities.
   * - **International Union of Basic and Clinical Pharmacology (IUPHAR)**
     - ``iuphar_pharmacologist``
     - Pharmacokinetics (PK/PD), drug clearance (:math:`\text{mL/min/kg}`), elimination half-life, receptor binding affinity (:math:`K_d`).
   * - **International Union of Physiological Sciences (IUPS)**
     - ``iups_physiologist``
     - Cardiac output (:math:`\text{L/min}`), glomerular filtration rate (:math:`\text{mL/min/1.73m}^2`), membrane potential (:math:`\text{mV}`), blood pressure (:math:`\text{mmHg}`).
   * - **International Union of Toxicology (IUTOX)**
     - ``iutox_toxicologist``
     - :math:`\text{LD}_{50}` (:math:`\text{mg/kg}`), :math:`\text{LC}_{50}` (:math:`\text{mg/m}^3`), acceptable daily intake (ADI), benchmark dose levels.
   * - **International Union of Nutritional Sciences (IUNS)**
     - ``iuns_nutritionist``
     - Dietary reference intakes, :math:`\text{kcal}` vs :math:`\text{kJ}`, retinol activity equivalents (:math:`\mu\text{g RAE}`), macronutrient energy densities.
   * - **International Union of Food Science and Technology (IUFoST)**
     - ``iufost_food_scientist``
     - Water activity (:math:`a_w`), pasteurization lethality (:math:`F_0`), thermal death time (:math:`D`-value), food rheology (Bostwick consistometer).
   * - **IUPESM / IOMP (Medical Physics)**
     - ``iomp_medical_physicist``
     - Absorbed radiation dose (Gray, :math:`\text{Gy}`), equivalent biological dose (Sievert, :math:`\text{Sv}`), Kerma-area product (:math:`\text{Gy}\cdot\text{cm}^2`).
   * - **IUPESM / IFMBE (Biomedical Engineering)**
     - ``ifmbe_biomedical_engineer``
     - Biosensor impedance (:math:`\Omega\cdot\text{cm}^2`), physiological transducers, IEEE 11073 / UCUM medical device metrics.

5. Social, Behavioral & Human Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Scientific Union / Organization
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Psychological Science (IUPsyS)**
     - ``iupsys_psychologist``
     - Reaction time latency (:math:`\text{ms}`), psychophysical thresholds (JND, :math:`\text{dB}`), psychometric :math:`z`-scores, Likert scale standardization.
   * - **International Sociological Association (ISA)**
     - ``isa_sociologist``
     - Socio-economic status indexes, Gini inequality coefficients, demographic rates per 1,000 population, survey weight normalizations.
   * - **International Union for the Scientific Study of Population (IUSSP)**
     - ``iussp_demographer``
     - Total fertility rates, age-specific mortality rates, life expectancy at birth, net migration rates per 1,000 person-years.
   * - **World Anthropological Union (WAU)**
     - ``wau_anthropologist``
     - Osteometric dimensions (:math:`\text{mm}`), radiocarbon dating (calibrated years BP), customary and indigenous ethno-metrology systems.
   * - **Society for Social Studies of Science (4S)**
     - ``four_s_science_studies``
     - Sociotechnical infrastructure of SI redefinition, institutional standardization rituals, metrological conventions in policy.
   * - **International Union for History and Philosophy of Science and Technology (IUHPST)**
     - ``iuhpst_historian_philosopher``
     - Epistemology of measurement, operationalism vs realism in SI base units, 1875 Metre Convention, 2019 quantum SI redefinition.

Persona Reporting & Distribution Analytics
-------------------------------------------

During every generation cycle, DRUM-ML automatically aggregates persona generation statistics and generates a dedicated **Persona Distribution Report** at ``dataset/persona_report.md`` and ``dataset/persona_report.json``.

You can also generate and view this report on-demand using the CLI:

.. code-block:: bash

   drum-ml persona-report

This generates:

* **Disciplinary Sector Breakdown:** Percentage and count of samples across Physical, Bio-Medical, Earth/Space, Social, and Standards sectors.
* **Per-Persona Sample Counts:** Exact sample tally and balance metrics across all 39 scientific unions.
* **Archetype Interaction Matrix:** Cross-tabulation showing how each persona intersects with the 6 pedagogical archetypes.

Configuring Personas in ``config.yaml``
---------------------------------------

By default, all 39 personas are active in the pipeline. To target specific scientific unions or run domain-adapted subsets:

.. code-block:: yaml

   augmenter:
     enabled: true
     provider: "openai"
     model: "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
     api_base: "http://localhost:8000/v1"
     api_key: "d2855fd4ce293b5e91aae58ae425cd09f7229cb8c97810f597a75a20848e6d69"
     concurrency_limit: 60
     temperature: 0.7
     variations_per_archetype: 2
     personas:
       # Run specific disciplinary unions:
       - "nist_metrologist"
       - "iupac_chemist"
       - "iau_astronomer"
       - "iucr_crystallographer"
       - "iugg_geodesist_geophysicist"
       - "iomp_medical_physicist"
       - "firmware_iot_engineer"

