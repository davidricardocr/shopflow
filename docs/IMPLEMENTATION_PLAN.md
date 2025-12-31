# ShopFlow Implementation Plan

> **Technical Assessment - E-commerce Returns Prediction**  
> Follow this plan sequentially to ensure quality, completeness, and adherence to requirements.

---

## 🎯 Key Principles

| Principle | Application |
|-----------|-------------|
| **Clean Architecture** | Separate concerns: data, domain logic, evaluation, presentation |
| **Strategy Pattern** | Interchangeable model strategies for easy experimentation |
| **Factory Pattern** | Model and preprocessor creation |
| **Single Responsibility** | Each module does ONE thing well |
| **DRY (Don't Repeat Yourself)** | Reusable evaluation and preprocessing pipelines |
| **Type Hints** | All functions with proper type annotations |
| **Docstrings** | Google-style docstrings for all public functions |

---

## 📚 Advanced Libraries to Use

```python
# Core
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3

# Advanced Evaluation & Visualization
matplotlib >= 3.7
seaborn >= 0.12
plotly >= 5.0              # Interactive threshold analysis

# Model Improvement
optuna >= 3.0              # Bayesian hyperparameter optimization
xgboost >= 2.0             # Gradient boosting (if justified)
lightgbm >= 4.0            # Alternative boosting

# Code Quality
pydantic >= 2.0            # Data validation & config management
loguru                     # Elegant logging

# Serialization
joblib                     # Model persistence
```

---

## 📁 Final Repository Structure

```
shopflow/
├── data/
│   ├── ecommerce_returns_train.csv
│   └── ecommerce_returns_test.csv
│
├── docs/
│   ├── shopflow_project.md          # Challenge instructions
│   └── IMPLEMENTATION_PLAN.md       # This file
│
├── models/                           # Generated artifacts
│   ├── baseline_model.pkl
│   ├── final_model.pkl              # DELIVERABLE
│   ├── scaler.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── mora_david_challenge.ipynb   # MAIN DELIVERABLE
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings & constants
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                # Data loading utilities
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Preprocessing pipeline
│   │   └── transformers.py          # Custom transformers
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py           # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract base model (Strategy)
│   │   ├── baseline.py              # Logistic regression baseline
│   │   ├── improved.py              # Improved model(s)
│   │   └── factory.py               # Model factory
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Custom business metrics
│   │   ├── threshold.py             # Threshold optimization
│   │   └── visualization.py         # Plots and charts
│   └── business/
│       ├── __init__.py
│       └── cost_analysis.py         # Cost-benefit calculations
│
├── tests/                            # Optional but impressive
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md                         # Updated with usage instructions
├── baseline_model.py                 # Original script (reference)
├── requirements.txt
└── summary.md                        # DELIVERABLE - Executive Summary
```

---

## ✅ Implementation Checklist

### Phase 0: Project Setup
- [ ] Create folder structure (`models/`, `notebooks/`, `src/`)
- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `src/config.py` with business constants
- [ ] Update `README.md` with project overview
- [ ] Commit: `"Setup project structure and dependencies"`

### Phase 1: Core Infrastructure (src/)
- [ ] `src/config.py` - Pydantic config with costs ($18, $3, 35%)
- [ ] `src/data/loader.py` - Data loading with validation
- [ ] `src/preprocessing/transformers.py` - Custom sklearn transformers
- [ ] `src/preprocessing/pipeline.py` - Full preprocessing pipeline
- [ ] Commit: `"Add core infrastructure with preprocessing pipeline"`

### Phase 2: Evaluation Framework
- [ ] `src/evaluation/metrics.py` - Business-aligned metrics
  - Expected Value per customer
  - Cost-weighted accuracy
  - Category-specific metrics
- [ ] `src/evaluation/threshold.py` - Threshold optimization
- [ ] `src/evaluation/visualization.py` - Confusion matrix, ROC, threshold curves
- [ ] `src/business/cost_analysis.py` - Financial impact calculator
- [ ] Commit: `"Add evaluation framework with business metrics"`

### Phase 3: Model Architecture
- [ ] `src/models/base.py` - Abstract base class (Strategy Pattern)
- [ ] `src/models/baseline.py` - Baseline logistic regression
- [ ] `src/models/improved.py` - Improved model implementations
- [ ] `src/models/factory.py` - Model factory
- [ ] Commit: `"Add model architecture with Strategy pattern"`

### Phase 4: Jupyter Notebook (MAIN DELIVERABLE)
- [ ] Create `notebooks/mora_david_challenge.ipynb`
- [ ] **Part 1: Baseline Evaluation** (10 min allocation)
  - [ ] Load baseline model
  - [ ] Multiple metrics with justification (≥3 metrics)
  - [ ] Confusion matrix with business interpretation
  - [ ] Performance by product category
  - [ ] Weakness identification with evidence
  - [ ] Markdown explanations for each finding
- [ ] **Part 2: Business-Aligned Metrics** (20 min allocation)
  - [ ] Define "success" in business terms
  - [ ] 2-3 recommended metrics with justification
  - [ ] Cost-benefit analysis with formulas
  - [ ] Threshold optimization with visualization
  - [ ] "Good enough to deploy" criteria
  - [ ] Financial impact quantification
- [ ] **Part 3: Model Improvement** (20 min allocation)
  - [ ] Document hypothesis BEFORE each experiment
  - [ ] Feature engineering with reasoning
  - [ ] Hyperparameter tuning (Optuna)
  - [ ] Validate no data leakage
  - [ ] Quantified comparison vs baseline
  - [ ] Overfitting prevention evidence
  - [ ] What worked AND what didn't
- [ ] **Part 4: Deployment Planning** (10 min allocation)
  - [ ] Monitoring metrics specification
  - [ ] Model degradation detection strategy
  - [ ] Specific alert thresholds
  - [ ] Retraining triggers
  - [ ] Rollback criteria
  - [ ] A/B testing strategy
  - [ ] Seasonal pattern detection approach
- [ ] Run all cells and verify outputs visible
- [ ] Commit: `"Add complete challenge notebook with all 4 parts"`

### Phase 5: Deliverables Finalization
- [ ] Save `models/model.pkl` (final trained model)
- [ ] Create `summary.md` (300-500 words)
  - [ ] Approach overview
  - [ ] Key findings
  - [ ] Business impact estimate ($ terms)
  - [ ] Deployment recommendation
- [ ] Update `README.md` with:
  - [ ] Quick start instructions
  - [ ] Project structure explanation
  - [ ] Results summary
- [ ] Commit: `"Add final model and executive summary"`

### Phase 6: Quality Assurance
- [ ] Verify all outputs visible in notebook
- [ ] Check no red flags from rubric:
  - [ ] ✓ Multiple metrics (not just accuracy)
  - [ ] ✓ Class imbalance discussion
  - [ ] ✓ Cost-benefit analysis included
  - [ ] ✓ Baseline comparison for all improvements
  - [ ] ✓ Validation strategy documented
  - [ ] ✓ Feature choices explained
  - [ ] ✓ Overfitting prevention discussed
  - [ ] ✓ Specific deployment plan (not vague)
  - [ ] ✓ Data drift consideration
  - [ ] ✓ Business impact in summary
- [ ] Push to GitHub
- [ ] Verify repository is PUBLIC

---

## 🔑 Critical Implementation Notes

### Business Constants (MUST USE)
```python
RETURN_COST = 18.0          # Cost when item is returned
INTERVENTION_COST = 3.0      # Cost of intervention
INTERVENTION_EFFECT = 0.35   # Reduces return probability by 35%
SAVINGS_PER_TP = 15.0        # $18 - $3 = $15 saved
```

### Expected Value Formula
```python
EV = (TP_rate × $15) - (FP_rate × $3) - (FN_rate × $18)
```

### Metrics to Implement
1. **Primary**: Precision at optimal threshold
2. **Secondary**: Expected Value per customer
3. **Segment**: Recall for Fashion category (30% return rate vs 15% overall)

### Threshold Analysis
- Default: 0.5
- Optimize for: Maximum Expected Value
- Report: Precision, Recall, EV, Monthly Savings at each threshold

### Deployment Criteria (from example)
- ✓ EV > $2.00/customer
- ✓ Precision > 0.65
- ✓ Recall > 0.50
- ✓ Passes A/B test (95% confidence)

---

## 💡 Design Patterns to Showcase

### 1. Strategy Pattern (Models)
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return probability estimates."""
        pass
```

### 2. Factory Pattern (Model Creation)
```python
class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create(model_type: str, **kwargs) -> BaseModel:
        models = {
            "baseline": BaselineModel,
            "optimized_lr": OptimizedLogisticRegression,
            "xgboost": XGBoostModel,
        }
        return models[model_type](**kwargs)
```

### 3. Pipeline Pattern (Preprocessing)
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline() -> Pipeline:
    """Create a reusable preprocessing pipeline."""
    ...
```

---

## ⚠️ Avoid These Red Flags

| Red Flag | How to Avoid |
|----------|--------------|
| Only accuracy | Report 3+ metrics with business justification |
| No class imbalance discussion | Analyze target distribution, discuss impact |
| No cost-benefit analysis | Use $18/$3 costs, calculate EV |
| Improvements without baseline | Always show delta vs baseline |
| No validation strategy | Use cross-validation, hold-out set |
| Can't explain features | Document WHY for each feature |
| No overfitting discussion | Show train vs test performance |
| Vague deployment plan | Specific metrics, thresholds, triggers |
| No data drift consideration | Document monitoring strategy |
| Summary lacks business impact | Include $ estimates, ROI |

---

## 📊 Notebook Section Templates

### Part 1 Header
```markdown
# Part 1: Baseline Evaluation

## Objectives
- Evaluate the provided logistic regression baseline
- Identify strengths and weaknesses
- Establish performance benchmarks

## Key Questions to Answer
1. What are the model's strengths and weaknesses?
2. Where does it fail most? (by category, by customer segment)
3. Is accuracy the right metric? Why or why not?
```

### Part 2 Header
```markdown
# Part 2: Business-Aligned Metrics

## Financial Context
- Return cost: $18 per item
- Intervention cost: $3 (reduces return probability by 35%)
- Goal: Maximize ROI while catching high-risk returns

## Key Deliverables
1. Business success definition
2. 2-3 recommended metrics with justification
3. Optimal threshold selection
4. "Good enough to deploy" criteria
```

### Part 3 Header
```markdown
# Part 3: Model Improvement

## Methodology
For each improvement attempt:
1. **Hypothesis**: What we expect and why
2. **Implementation**: The actual change
3. **Validation**: Evidence it works and doesn't overfit
4. **Result**: Quantified comparison to baseline
```

### Part 4 Header
```markdown
# Part 4: Deployment Planning

## Production Monitoring Plan
Specific metrics, thresholds, and triggers for a production ML system.

## Stakeholder Summary
Business-friendly summary with ROI estimates.
```

---

## 🚀 Execution Order

```
1. Phase 0 → Commit
2. Phase 1 → Commit
3. Phase 2 → Commit
4. Phase 3 → Commit
5. Phase 4 → Commit (largest, most important)
6. Phase 5 → Commit
7. Phase 6 → Push to GitHub
```

**Total estimated time**: 90 minutes (as per challenge requirements)

---

## 📝 Final Checklist Before Submission

- [ ] Notebook runs from top to bottom without errors
- [ ] All cell outputs are visible
- [ ] No hardcoded paths (use relative paths or config)
- [ ] `summary.md` is 300-500 words
- [ ] `model.pkl` is saved and loadable
- [ ] Repository is PUBLIC on GitHub
- [ ] README has clear instructions
- [ ] All commits have meaningful messages

---

*Good luck! Focus on quality over quantity. Thorough Parts 1-2 > rushed all 4.*
