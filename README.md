# Atmospheric Processes and Glacier Mass Balance in Northeast Greenland

**Repository:** `shahi_et_al_2027_atmos`

This repository contains the Python codes, data-processing workflows, and figures associated with the manuscript:

> Shahi et al. (2027). *[Timescales of sea ice–glacier coupling in Northeast Greenland: Freya Glacier as a case study].*

## 1. Repository overview

The repository is organized into three main directories:

- `codes/`: Python scripts and modules used for data processing, analysis, and visualization.
- `data/`: Raw and processed datasets used in the analysis.
- `figures/`: Figures generated for the manuscript and supplementary material.

## 2. Repository structure

```text
shahi_et_al_2027_atmos/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml
│
├── codes/
│   ├── core/
│   │   └── [Main analysis scripts]
│   │
│   ├── modules/
│   │   └── [Analysis and processing modules]
│   │
│   └── utilities/
│       └── [Helper functions and utilities]
│
├── data/
│   ├── README.md
│   ├── raw_data/
│   │   └── [Original input datasets]
│   │
│   └── processed_data/
│       └── [Processed datasets]
│
└── figures/
    ├── manuscript_figures/
    │   └── [Main manuscript figures]
    │
    └── supplementary_figures/
        └── [Supplementary figures]
```

## 3. Software requirements

The analysis was performed using Python.

The required Python packages and their versions are documented in `environment.yml`.

### Creating the Python environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate shahi_atmos
```

To update the environment:

```bash
conda env update -f environment.yml --prune
```

## 4. Data availability

The `data/` directory contains the datasets used in the analysis.

- `raw_data/`: Original datasets obtained from external data providers.
- `processed_data/`: Datasets generated through preprocessing and subsequent analysis.

Please refer to `data/README.md` for descriptions of the datasets, their original sources, access information, and processing details.

Datasets that are publicly available from external repositories are referenced in the data documentation. Large datasets or data subject to third-party restrictions may not be redistributed directly through this repository.

## 5. Code description

The `codes/` directory contains the Python scripts used throughout the analysis.

- `core/`: Main scripts controlling the analysis workflow.
- `modules/`: Reusable modules for specific analytical procedures.
- `utilities/`: Supporting functions, helper scripts, and common utilities.

The analysis scripts use the modules and utilities to process the input data and generate the results presented in the manuscript.

## 6. Reproducing the analysis

To reproduce the analysis:

1. Clone or download this repository.
2. Install the required Python environment using `environment.yml`.
3. Obtain the input datasets listed in `data/README.md`.
4. Place the datasets in their corresponding directories under `data/`.
5. Execute the analysis scripts in `codes/core/` in the documented order.
6. The resulting figures and processed datasets will be generated in the corresponding output directories.

**Note:** Update this section with the exact script names, execution order, and required input files before publication.

## 7. Figures

The `figures/` directory contains the figures generated for the manuscript.

- `manuscript_figures/`: Main figures included in the manuscript.
- `supplementary_figures/`: Figures included in the supplementary material.

## 8. Citation

If you use the codes or datasets provided in this repository, please cite the associated manuscript and software repository.

**Manuscript:**

Shahi et al. (2027). *[Insert manuscript title].* [Journal name]. [DOI]

**Software repository:**

Shahi et al. (2027). *Code and data for [short manuscript title].* GitHub repository. https://doi.org/[Zenodo DOI]

## 9. License

The code is distributed under the license specified in the `LICENSE` file. Please refer to the license for terms of use and redistribution.

## 10. Contact

For questions regarding the code, data processing, or analysis, please contact:

Sonika Shahi

[Insert institutional email address]