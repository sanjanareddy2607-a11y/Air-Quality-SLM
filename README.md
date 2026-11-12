# Air Quality SLM
### Domain-Specific Small Language Model for Air Quality Data

**Air Quality SLM** is a research and development project focused on exploring the development of a domain-specific Small Language Model (SLM) for the air-quality domain.

The project uses structured air-quality measurements and domain-related context to build a foundation for preparing, processing, and eventually training and evaluating a language model specialised in air-quality information.

The objective is to explore how language models can be adapted to a focused technical domain rather than relying entirely on general-purpose language models.

---

## Project Overview

Air-quality information involves multiple pollutants, environmental measurements, timestamps, sensor readings, and domain-specific terminology. Working with this information requires careful data preparation and meaningful representation.

This project follows a structured workflow:

1. **Data Collection** – Obtain air-quality data from the UCI Machine Learning Repository.
2. **Data Exploration** – Examine the dataset, its features, structure, and missing values.
3. **Data Preprocessing** – Clean and transform the original data into a usable format.
4. **Dataset Preparation** – Organise the data into training, validation, and testing subsets.
5. **Context and Knowledge Preparation** – Collect domain-related information and feature context.
6. **Tokenization** – Prepare text data for language-model processing.
7. **Model Development** – Explore the development and adaptation of a domain-specific SLM.
8. **Evaluation** – Assess model performance using appropriate evaluation methods.

*Note: The project is being developed incrementally. The status of each stage should be updated as implementation progresses.*

---

## Dataset

The primary structured dataset used in this project is the **Air Quality dataset** from the UCI Machine Learning Repository.

**Original dataset:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

### Dataset information

| Property | Description |
|---|---|
| Dataset | UCI Air Quality |
| Original observations | 9,471 |
| Processed observations | 9,357 |
| Processed features | 15 |
| Data type | Time-series environmental measurements |
| Data domain | Air quality and environmental monitoring |

### Data preprocessing

The preprocessing workflow includes:

- Reading and interpreting the original dataset.
- Handling semicolon-separated values and decimal-comma notation.
- Removing empty trailing columns and blank records.
- Converting the dataset's `-200` missing-value sentinel into `NaN`.
- Preserving missing observations rather than silently replacing them.
- Maintaining chronological ordering.
- Splitting the processed dataset into training, validation, and testing sets.

### Dataset split

| Subset | Records | Proportion |
|---|---:|---:|
| Training | 6,549 | 70% |
| Validation | 936 | 10% |
| Testing | 1,872 | 20% |
| **Total** | **9,357** | **100%** |

The split files are intended to support model development, validation, and final testing.

---

## Repository Structure

The repository is organised into separate directories to keep datasets, notebooks, source code, research material, and documentation manageable.

```text
Air-Quality-SLM/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── training/
│
├── notebooks/
│   ├── 0_gpu_check.ipynb
│   └── Data-Convertion.ipynb
│
├── src/
│   ├── preprocess_air_quality.py
│   ├── validate_dataset.py
│   ├── package_dataset.py
│   └── upload_to_huggingface.py
│
├── research/
│   ├── Air_Quality_50_Feature_Context_PDFs/
│   └── Feature3.3.md
│
├── models/
│
├── docs/
│   └── VALIDATION_REPORT.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

*The tree above illustrates the intended organisation. Update it if the actual repository contains additional files or different directory names.*

---

## Technologies and Tools

The project uses or explores the following technologies:

- **Python** – Data processing and project implementation.
- **Jupyter Notebook** – Data exploration and experimentation.
- **Pandas and NumPy** – Structured data handling and numerical operations.
- **Hugging Face** – Dataset hosting and language-model development workflows.
- **Transformer-based architectures** – The foundation for exploring language-model development.
- **Git and GitHub** – Version control and project documentation.

The exact dependencies and versions are maintained in the project's dependency file.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sanjanareddy2607-a11y/Air-Quality-SLM.git
cd Air-Quality-SLM
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Explore the project

Open the notebooks in the `notebooks/` directory to explore the data preparation and experimentation workflow.

Run the relevant preprocessing and validation scripts from `src/` according to their documented usage.

---

## Project Objectives

- Prepare a reliable and reproducible air-quality dataset.
- Organise domain-specific information into a structured format.
- Explore suitable tokenization strategies for air-quality-related text.
- Investigate the development of a compact, domain-specific language model.
- Evaluate the model's ability to interpret and respond to air-quality-related information.
- Document the complete workflow to support reproducibility and future development.

---

## Current Development Status

This repository is an evolving research and development project.

| Component | Status |
|---|---|
| Original dataset acquisition | Completed |
| Initial data preprocessing | Completed |
| Training, validation and test split | Completed |
| Dataset validation and packaging | Prepared |
| Domain-context collection | In progress |
| Tokenization pipeline | Upcoming |
| Model training / adaptation | Upcoming |
| Model evaluation | Upcoming |

Update this table as each component is implemented and verified.

---

## Dataset and References

The primary dataset is sourced from:

- Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine.
- [Air Quality Dataset – UCI Repository](https://archive.ics.uci.edu/dataset/360/air+quality)

Please consult the original dataset page for its citation and usage terms.

---

## Future Work

Potential next stages include:

- Developing and comparing tokenization approaches.
- Preparing domain-specific text and instruction data.
- Experimenting with compact Transformer-based architectures.
- Training or adapting a suitable SLM.
- Evaluating model performance and domain understanding.
- Improving reproducibility and documenting experimental findings.

---

## Author

**Sanjana Reddy**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: [@sanjanareddy2607-a11y](https://github.com/sanjanareddy2607-a11y)

---

*This project is developed for academic and research purposes as an exploration of domain-specific language models for the air-quality domain.*
