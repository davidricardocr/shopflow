# ShopFlow - E-commerce Returns Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Machine learning solution for predicting e-commerce product returns, with business-aligned evaluation metrics and cost-benefit optimization.

**Challenge Focus:**
- Evaluate and improve a baseline logistic regression model
- Define business-aligned metrics (cost-benefit analysis)
- Optimize decision thresholds for ROI maximization
- Create production-ready deployment strategy

## Quick Start

```bash
# Clone repository
git clone https://github.com/davidricardocr/shopflow.git
cd shopflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run baseline model
python baseline_model.py
```

## Project Structure

```
shopflow/
├── data/                    # Train and test datasets
├── docs/                    # Challenge documentation
├── models/                  # Saved model artifacts
├── notebooks/               # Jupyter notebooks (main deliverable)
├── src/                     # Source code modules
│   ├── config.py           # Configuration and business constants
│   ├── data/               # Data loading utilities
│   ├── preprocessing/      # Feature preprocessing pipeline
│   ├── features/           # Feature engineering
│   ├── models/             # Model implementations
│   ├── evaluation/         # Metrics and visualization
│   └── business/           # Cost-benefit analysis
├── baseline_model.py        # Original baseline script
├── requirements.txt         # Python dependencies
└── summary.md              # Executive summary (deliverable)
```

## Business Context

| Parameter | Value |
|-----------|-------|
| Return Cost | $18 per item |
| Intervention Cost | $3 per customer |
| Intervention Effect | 35% reduction in return probability |
| True Positive Value | $15 saved ($18 - $3) |

## Key Deliverables

1. **Jupyter Notebook** - Complete analysis with all 4 parts
2. **Executive Summary** - Business-focused 300-500 word summary
3. **Final Model** - Trained model artifact (`.pkl`)

## Results

*Results will be added after completing the challenge.*

## Author

David Mora

---

*Built for ShopFlow Technical Assessment*
