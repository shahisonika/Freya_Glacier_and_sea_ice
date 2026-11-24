## 1. Raw data

- `station_data/`:
    -Freya Glacier’s automated weather station dataset was provided by the Programme for Monitoring of the Greenland Ice Sheet (PROMICE) and is available via the PROMICE data portal (https://promice.org/download-data/)
    -The Zackenberg Research Station data from the Greenland Ecosystem Monitoring program was provided by Asiaq, Greenland Survey (Greenland), data available from https://data.g-e-m.dk/.
- `masks/`: Raw and processed datasets used in the analysis.
- `complex_mca_input/`: The Regional Atmospheric Climate Model (RACMO) dataset was provided by Brice Noël from IMAU used by the input of complex MCA.

## 2. Processed data

- `complex_xeofs_results/`: Results of complex MCA
- `extracted_racmo_var/`: Varaibles interpolated at the specific location in a gridded files
- `sic_phenology/`: Results of sea ice phenology, its trend, and related statistics


# data/ Directory

This directory contains the raw and processed datasets used in the analysis.

## 1. raw_data/

The `raw_data/` directory contains the original datasets obtained from external data providers.

- **`station_data/`**
  Contains meteorological observations from the following stations:
  - **Freya Glacier Automated Weather Station (AWS):** Data provided by the Programme for Monitoring of the Greenland Ice Sheet (PROMICE). Available through the [PROMICE Data Portal](https://promice.org/download-data/).
  - **Zackenberg Research Station:** Data obtained through the Greenland Ecosystem Monitoring (GEM) programme and provided by Asiaq, Greenland Survey. Available via the [GEM Data Portal](https://data.g-e-m.dk/).

- **`masks/`**
  Contains the raw and processed spatial masks used in the analysis.

- **`complex_mca_input/`**
  Contains the Regional Atmospheric Climate Model (RACMO) datasets provided by Brice Noël (Institute for Marine and Atmospheric Research Utrecht, IMAU), used as input for the complex Maximum Covariance Analysis (MCA).

## 2. Processed Data

This directory contains datasets and outputs generated during the data-processing and analysis workflows.

- **`complex_xeofs_results/`**
  Contains the results of the complex MCA performed using the `xeofs` package.

- **`extracted_racmo_var/`**
  Contains RACMO variables extracted and interpolated at the selected study locations from the original gridded datasets.

- **`sic_phenology/`**
  Contains derived sea-ice phenology metrics, associated temporal trends, and statistical analysis results.
