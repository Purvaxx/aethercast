# AetherCast
### Universal Time-Series Forecasting Platform

AetherCast AI is a machine-learning-driven forecasting platform designed to bring dataset exploration, forecasting, model-driven analysis, and reporting into a unified application.

The project focuses on simplifying the forecasting workflow while retaining control over dataset preparation, forecasting configuration, prediction analysis, and result export.

## Overview

Time-series forecasting is often fragmented across separate tools for data preparation, model execution, visualization, and reporting. AetherCast AI brings these stages together in a single interactive application, providing a consistent workflow from raw data to forecast results.

The platform combines a Python-based application layer with forecasting and machine-learning components to support configurable prediction workflows and interpretable result exploration.

## Key Capabilities

- **Dataset Ingestion and Inspection** — Upload time-series datasets and examine their structure before forecasting.
- **Configurable Forecasting** — Configure forecasting parameters according to the dataset and prediction requirements.
- **Machine Learning Model Integration** — Use forecasting models within a unified workflow for generating predictions.
- **Forecast Visualization** — Examine predicted trends and compare forecast outputs with available historical observations.
- **Analytical Insights** — Explore forecast results through visual summaries and supporting analysis.
- **Report Generation** — Export forecast outputs as CSV files and downloadable reports.
- **Forecast Run Management** — Access saved forecast runs through the application's Insights and Reports sections, subject to the deployment environment's storage limitations.
- **Interactive Interface** — Navigate the forecasting workflow through a dedicated multi-page Streamlit application.

## Technical Architecture

AetherCast AI is organized around the following functional layers:

| Layer | Responsibility |
|---|---|
| User Interface | Dataset upload, forecasting configuration, navigation, and result presentation |
| Data Processing | Dataset inspection, validation, and preprocessing |
| Forecasting Engine | Execution of forecasting models and generation of predictions |
| Analysis Layer | Visualization and exploration of forecast outputs |
| Reporting Layer | Export of prediction data and generated reports |
| Application Storage | Management of saved forecast runs and associated outputs |

The forecasting engine and application interface are treated as distinct responsibilities, allowing the user-facing workflow to evolve without unnecessarily replacing the underlying forecasting implementation.

## Machine Learning and Model Development

The project incorporates forecasting and machine-learning components into a common application workflow.

The model-development layer is intended to support the relationship between historical observations, forecasting configurations, and generated predictions. Model selection, preprocessing, and evaluation remain important parts of the overall forecasting process.

Language-model functionality, where implemented, is a separate component from numerical time-series forecasting. The specific model architectures, training procedures, and responsibilities should be documented alongside the relevant implementation.

> Model architectures, training datasets, evaluation metrics, and comparative results should be reported only where supported by the project's implementation and experimental records.

## Application Workflow

1. **Upload** — Provide a time-series dataset.
2. **Inspect** — Review the dataset and its structure.
3. **Configure** — Set up the forecasting task and relevant parameters.
4. **Forecast** — Execute the forecasting pipeline.
5. **Analyze** — Review predictions and explore available insights.
6. **Export** — Download forecast data and reports for further analysis.

## Technology Stack

- **Programming Language:** Python
- **Application Framework:** Streamlit
- **Core Domain:** Time-Series Forecasting and Machine Learning
- **Data Analysis:** Dataset Processing and Forecast Visualization
- **Interface Rendering:** Streamlit with custom web-based visual effects
- **Output Formats:** CSV and downloadable reports

The complete dependency list is maintained in `requirements.txt`.

## Running Locally

### Prerequisites

- Python version compatible with the project's dependencies
- Git
- A virtual environment

### Installation

Clone the repository:

```bash
git clone https://github.com/Purvaxx/aethercast.git
cd aethercast
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application using the project's launcher:

```powershell
.venv\Scripts\python.exe universal_app\launch.py
```

Open the local address printed by the application in your browser. The current local setup uses port `8503`.

## Project Demonstration

A recorded demonstration video will be added to this section.

The video will showcase the application's workflow, from dataset upload and forecasting configuration to prediction visualization, insights, and report generation.

**Demo video:**

https://github.com/user-attachments/assets/50f668c1-b8c9-4567-8470-892a93e9ae24

## Deployment

The application is designed for deployment using Streamlit Community Cloud, subject to dependency compatibility, hosting resource limits, and the configuration of the deployment environment.

Repository privacy and application access permissions are separate considerations. Persistent forecast history also depends on the storage available in the hosted environment.

## Project Status

AetherCast AI is an evolving forecasting application. Its capabilities, model implementations, and evaluation results should be assessed against the current codebase and documented experiments.

## Repository

**GitHub:** https://github.com/Purvaxx/aethercast

---

*Developed as a unified approach to dataset-driven forecasting, model integration, and analytical reporting.*
