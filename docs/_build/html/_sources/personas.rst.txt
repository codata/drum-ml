.. _scientific_unions_and_personas:

Scientific Unions & Domain Personas
===================================

The **DRUM-ML** pipeline integrates domain-specific persona profiles representing the **CODATA DRUM Scientific Unions and Standards Bodies**.

While **Agent 2 (Archetype Scaffolder)** deterministically derives ground-truth answers from the BIPM, CODATA, and QUDT master knowledge graphs, **Agent 3 (Linguistic Diversity Augmenter)** uses these personas to generate realistic questions from diverse disciplinary perspectives.

Overview of Union & Standards Coverage
---------------------------------------

The 35 available personas in DRUM-ML are organized into 5 major disciplinary sectors:

1. Standards Bodies & Global Science Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Scientific Body / Organization
     - Persona Key (``config.yaml``)
     - Metrological Focus & Domain Coverage
   * - **National Institute of Standards and Technology (NIST)**
     - ``nist_metrologist``
     - NIST SP 330/811 guidelines, Kibble balance, optical clocks, primary calibration standards, fundamental constants.
   * - **National Research Council Canada (NRC)**
     - ``nrc_metrologist``
     - Quantum Hall resistance, Josephson voltage standards, mass dissemination, precision electrical metrology.
   * - **International Science Council (ISC)**
     - ``isc_science_policy``
     - Interdisciplinary science data interoperability, FAIR digital representation of units, global scientific cooperation.
   * - **Academic Metrologist (BIPM / NMI)**
     - ``academic_metrologist``
     - BIPM 9th Edition SI brochure, VIM3 terminology, exact SI 2019 defining constants, LaTeX formulas.

2. Physical, Chemical & Mathematical Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Scientific Union
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Pure and Applied Physics (IUPAP)**
     - ``iupap_physicist``
     - IUPAP SUNAMCO symbols, base units, physical dimensions, SI realizations.
   * - **International Union of Pure and Applied Chemistry (IUPAC)**
     - ``iupac_chemist``
     - IUPAC Green Book, amount-of-substance (mol, mol/L), molar gas constant, standard atomic weights.
   * - **International Astronomical Union (IAU)**
     - ``iau_astronomer``
     - Parsecs, light-years, astronomical units, solar masses (:math:`M_\odot`), Jansky (:math:`\text{Jy}`) spectral flux.
   * - **International Union of Crystallography (IUCr)**
     - ``iucr_crystallographer``
     - Unit cell dimensions (:math:`\text{Å}`, :math:`\text{pm}`), reciprocal lattice vectors, electron density (:math:`e/\text{Å}^3`).
   * - **International Mathematical Union (IMU)**
     - ``imu_mathematician``
     - Dimensional vector spaces, Lie group symmetries in physics, Buckingham :math:`\Pi` theorem proofs.
   * - **Union Radio Scientifique Internationale (URSI)**
     - ``ursi_radio_scientist``
     - RF telemetry, antenna gain (:math:`\text{dBi}`), antenna noise temperature (:math:`\text{K}`), Poynting vectors.
   * - **International Union of Theoretical and Applied Mechanics (IUTAM)**
     - ``iutam_mechanics_engineer``
     - Stress tensors (:math:`\text{Pa}`), dynamic/kinematic viscosity, Reynolds/Mach/Nusselt numbers.
   * - **High-Energy Particle Physics**
     - ``particle_physicist``
     - Electronvolts (:math:`\text{eV/GeV/TeV}`), barn (:math:`\text{b}`) cross-sections, natural Planck units.
   * - **Aerospace & Propulsion**
     - ``aerospace_propulsion_engineer``
     - Specific impulse (:math:`I_{\text{sp}}` in seconds), Mach numbers, knots, dynamic pressure, hypersonic flow.

3. Earth, Space & Environmental Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Scientific Union
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Geodesy and Geophysics (IUGG)**
     - ``iugg_geodesist_geophysicist``
     - Gravitational anomalies (:math:`\text{mGal}`), geoid heights, seismic moments (:math:`\text{N}\cdot\text{m}`), geomagnetic field (:math:`\text{nT}`).
   * - **International Geographical Union (IGU)**
     - ``igu_geographer``
     - Spatial scale, map projections, geospatial attribute scaling, land area units (:math:`\text{ha}, \text{km}^2`).
   * - **International Society for Photogrammetry and Remote Sensing (ISPRS)**
     - ``isprs_photogrammetrist``
     - Ground sampling distance (:math:`\text{cm/px}`), spectral radiance, LiDAR point density (:math:`\text{pts/m}^2`).
   * - **International Society for Digital Earth (ISDE)**
     - ``isde_digital_earth``
     - Discrete Global Grid Systems (DGGS), multi-resolution planetary telemetry harmonization.
   * - **International Union of Soil Science (IUSS)**
     - ``iuss_soil_scientist``
     - Soil bulk density (:math:`\text{g/cm}^3`), cation exchange capacity (:math:`\text{cmol}(+)/\text{kg}`), hydraulic conductivity.
   * - **Climate & Energy Systems**
     - ``energy_environmental_scientist``
     - Atmospheric trace gases (:math:`\text{ppm, ppb}`), grid storage (:math:`\text{MWh, GWh}`), solar irradiance (:math:`\text{W/m}^2`).

4. Biological, Medical & Health Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Scientific Union
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Biological Sciences (IUBS)**
     - ``iubs_biologist``
     - Metabolic scaling rates (:math:`\text{W/kg}`), biomass density, population dynamics.
   * - **International Union of Immunological Societies (IUIS)**
     - ``iuis_immunologist``
     - International Units (:math:`\text{IU/mL}`) of biological activity, antibody titers, cytokine concentrations.
   * - **International Union of Basic and Clinical Pharmacology (IUPHAR)**
     - ``iuphar_pharmacologist``
     - Pharmacokinetics (PK/PD), drug clearance (:math:`\text{mL/min/kg}`), half-life, receptor binding affinity (:math:`K_d`).
   * - **International Union of Physiological Sciences (IUPS)**
     - ``iups_physiologist``
     - Cardiac output (:math:`\text{L/min}`), GFR (:math:`\text{mL/min/1.73m}^2`), membrane potential (:math:`\text{mV}`), blood pressure.
   * - **International Union of Toxicology (IUTOX)**
     - ``iutox_toxicologist``
     - :math:`\text{LD}_{50}` (:math:`\text{mg/kg}`), :math:`\text{LC}_{50}` (:math:`\text{mg/m}^3`), acceptable daily intake (ADI), threshold limits.
   * - **International Union of Nutritional Sciences (IUNS)**
     - ``iuns_nutritionist``
     - Dietary reference intakes, :math:`\text{kcal}` vs :math:`\text{kJ}`, retinol equivalents (:math:`\mu\text{g RAE}`), glycemic index.
   * - **International Union of Food Science and Technology (IUFoST)**
     - ``iufost_food_scientist``
     - Water activity (:math:`a_w`), pasteurization lethality (:math:`F_0`), thermal death time (:math:`D`-value), food rheology.
   * - **IUPESM / IOMP (Medical Physics)**
     - ``iomp_medical_physicist``
     - Absorbed radiation dose (Gray, :math:`\text{Gy}`), biological dose (Sievert, :math:`\text{Sv}`), Kerma product.
   * - **IUPESM / IFMBE (Biomedical Engineering)**
     - ``ifmbe_biomedical_engineer``
     - Biosensor impedance (:math:`\Omega\cdot\text{cm}^2`), physiological transducers, IEEE 11073 / UCUM device metrics.

5. Social, Behavioral & Human Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Scientific Union
     - Persona Key
     - Metrological Focus & Domain Coverage
   * - **International Union of Psychological Science (IUPsyS)**
     - ``iupsys_psychologist``
     - Reaction time latency (:math:`\text{ms}`), psychophysical thresholds (JND, :math:`\text{dB}`), psychometric :math:`z`-scores.
   * - **International Sociological Association (ISA)**
     - ``isa_sociologist``
     - Socio-economic status indexes, Gini inequality coefficients, demographic rates per 1,000 population.
   * - **International Union for the Scientific Study of Population (IUSSP)**
     - ``iussp_demographer``
     - Total fertility rates, infant mortality rates, life expectancy, net migration metrics.
   * - **World Anthropological Union (WAU)**
     - ``wau_anthropologist``
     - Cranial morphology (:math:`\text{mm}`), radiocarbon dating (years BP), customary ethno-metrology systems.
   * - **Society for Social Studies of Science (4S)**
     - ``four_s_science_studies``
     - Sociotechnical infrastructure of SI redefinition, metrological standardization rituals in society.
   * - **International Union for History and Philosophy of Science and Technology (IUHPST)**
     - ``iuhpst_historian_philosopher``
     - Epistemology of measurement, operationalism vs realism in SI base units, 1875 Metre Convention, 2019 quantum SI shift.

Configuring Personas in ``config.yaml``
---------------------------------------

To activate specific scientific unions in your dataset generation run:

.. code-block:: yaml

   augmenter:
     enabled: true
     provider: "ollama"
     model: "nemotron-3-nano:4b"
     personas:
       - "general_user"
       - "iupac_chemist"
       - "iau_astronomer"
       - "iucr_crystallographer"
       - "iugg_geodesist_geophysicist"
       - "iomp_medical_physicist"
