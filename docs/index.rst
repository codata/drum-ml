.. DRUM-ML documentation master file

DRUM-ML: Metrology & RDF-to-LLM Pipeline
=========================================

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/metrology-BIPM%20SI%209th%20Ed.-green.svg
   :target: https://si-digital-framework.org/SI
.. image:: https://img.shields.io/badge/constants-CODATA%20DRUM-orange.svg
   :target: https://github.com/codata/drum-constants
.. image:: https://img.shields.io/badge/Code%20License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by/4.0/

**DRUM-ML** (*Digital Representation of Units of Measurement for Machine Learning*) is an open scientific framework developed under the **CODATA DRUM (`Digital Representation of Units of Measurement <https://drum.codata.org>`_) Working Group**, led by **Pascal Heus** (``pascal@codata.org`` / ``drum@codata.org``).

It is an end-to-end, multi-agent autonomous framework designed to extract, synthesize, augment, validate, and package structured metrological knowledge into high-fidelity fine-tuning datasets for Large Language Models (LLMs).

By extracting semantic graphs from official international metrology authorities (**BIPM SI Digital Framework**, **CODATA DRUM Constants**, and **QUDT 2.1**) and verifying candidates through symbolic algebra and arbitrary-precision arithmetic, DRUM-ML eliminates unit-conversion hallucinations, dimensional errors, and constant misquotations in downstream AI models.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   objectives
   architecture
   methodology
   personas
   models
   assumptions
   training_and_eval
   api_reference

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
