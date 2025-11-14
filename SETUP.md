# Setup Guide for Student Grade Predictor

## System Requirements
- Python 3.7 or higher
- pip or conda package manager
- 100MB+ free disk space
- Internet connection (for first installation)

## Step-by-Step Installation

### 1. Navigate to Project Directory
```bash
cd "Student Grade Predictor"
```

### 2. Create Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, plotly, streamlit; print('All packages installed successfully!')"
```

## First Run

### Option A: Train Models + Launch Dashboard
```bash
python models/train.py
streamlit run app.py
```

### Option B: Direct Dashboard Launch (auto-trains models)
```bash
streamlit run app.py
```
Click "Train Models" button in the sidebar if models aren't found.

### Option C: Train Models Only
```bash
python models/train.py
```

## Troubleshooting Installation

### Problem: pip not found
**Windows:**
```bash
python -m pip install -r requirements.txt
```

### Problem: scikit-learn fails to install
```bash
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Problem: Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### Problem: Module import errors
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

## Generating New Training Data

The app automatically generates synthetic data. To regenerate:

```python
from utils.preprocess import generate_synthetic_data
df = generate_synthetic_data(n_samples=300)
df.to_csv('data/students.csv', index=False)
```

## Using Your Own Data

1. Prepare CSV file with columns: study_hours, attendance, previous_grade, sleep_hours, internet_usage, final_grade
2. Place in `data/students.csv`
3. Delete trained models:
```bash
del models\*.pkl
```
4. Run app - it will retrain with new data

## Environment Variables

Optional: Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[client]
showErrorDetails = false

[logger]
level = "info"
```

## Deactivating Virtual Environment
```bash
deactivate
```

## Next Steps

1. Read README.md for detailed documentation
2. Explore the Overview page for data insights
3. Use What-If Prediction to test scenarios
4. Consider deploying to Streamlit Cloud (see README.md)
