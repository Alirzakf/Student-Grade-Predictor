# Student Grade Predictor - OULAD Edition

A machine learning application using real UK public data to predict student outcomes.

## 🎯 Project Overview

This project uses the **Open University Learning Analytics Dataset (OULAD)** - a real, publicly available dataset from the UK's Open University - to predict student final results using machine learning.

**Dataset**: Open University Learning Analytics Dataset (OULAD)  
**Source**: https://analyse.kmi.open.ac.uk/open_dataset  
**License**: CC BY 4.0  
**Citation**: Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset. Scientific Data 4, Article number: 170005 (2017)

---

## 📊 Features & Data

### OULAD Features (Real Data from Open University)

**Demographic Features**:
- `age_band` - Age bracket (≤35, 35-55, >55)
- `gender` - Gender (M/F)
- `disability` - Disability status (Y/N)
- `region` - Geographic region in UK

**Academic Features**:
- `studied_credits` - Credits studied (30, 60, 90, 120)
- `num_of_prev_attempts` - Previous attempts at course
- `assessment_score_avg` - Average assessment score
- `days_since_registration` - Days since registration

**Engagement Features**:
- `clicks_total` - Total VLE (Virtual Learning Environment) clicks
- `activity_type` - Type of learning activity

### Target Variable
- `final_result` - Student outcome (Pass, Fail, Withdrawn, Distinction)

---

## 🚀 Getting Started

### 1. Get OULAD Data

**Option A: Download Real OULAD Data (Recommended)**
1. Visit: https://analyse.kmi.open.ac.uk/open_dataset
2. Register and download the dataset
3. Extract these CSV files to `data/` folder:
   - `students.csv`
   - `studentVle.csv`
   - `assessments.csv`
   - `studentAssessment.csv`

**Option B: Use Sample Data (For Quick Testing)**
The application automatically generates realistic OULAD-like sample data if CSV files are not found.

### 2. Installation

```bash
cd "Student Grade Predictor"
pip install -r requirements.txt
```

### 3. Run Dashboard

```bash
streamlit run app.py
```

Dashboard opens at: `http://localhost:8501`

---

## 🏗️ Project Structure

```
Student Grade Predictor/
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies
├── README.md                  # Main documentation
├── README_OULAD.md           # This file
├── data/
│   ├── students.csv          # OULAD students data
│   ├── studentVle.csv        # VLE interactions
│   ├── assessments.csv       # Assessment data
│   └── studentAssessment.csv # Assessment scores
├── models/
│   ├── train.py              # Training script
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
└── utils/
    ├── preprocess.py         # OULAD preprocessing
    ├── download_oulad.py     # OULAD loader
    └── visualize.py          # Visualizations
```

---

## 📋 Data Processing Pipeline

### Step 1: Data Loading
- Loads OULAD CSV files from `data/` directory
- Or generates realistic sample OULAD-like data

### Step 2: Feature Engineering
- Aggregates VLE interactions per student (clicks_total)
- Calculates mean assessment scores
- Computes days since registration
- Handles missing values (mean imputation)

### Step 3: Encoding
- **Categorical Features**: LabelEncoding (age_band, gender, disability, region)
- **Target Variable**: LabelEncoding (Pass, Fail, Withdrawn, Distinction)

### Step 4: Scaling
- **Numeric Features**: StandardScaler (studied_credits, num_of_prev_attempts, etc.)
- **Categorical Features**: One-hot or label encoded

### Step 5: Train/Test Split
- 80% training, 20% testing
- Stratified split (preserves class distribution)

---

## 🤖 Machine Learning Models

### Model 1: Logistic Regression Classifier
- **Type**: Multi-class classification
- **Best For**: Interpretability, understanding feature impacts
- **Hyperparameters**: max_iter=1000, multi_class='multinomial'

### Model 2: Random Forest Classifier
- **Type**: Ensemble (100 decision trees)
- **Best For**: Accuracy, non-linear relationships
- **Hyperparameters**: max_depth=15, min_samples_split=5

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many correct
- **Recall**: Of actual positives, how many detected
- **F1 Score**: Harmonic mean of precision & recall

---

## 📈 Dashboard Features

### Overview Page
- Dataset statistics (total students, features, class distribution)
- Raw data preview
- Class distribution chart
- Feature importance (Random Forest)
- Logistic regression coefficients

### What-If Prediction Page
Interactive controls to explore predictions:
- **Age Band** dropdown
- **Gender** dropdown
- **Disability** dropdown
- **Studied Credits** slider
- **Previous Attempts** slider
- **Average Assessment Score** slider
- **Days Since Registration** slider
- **Region** dropdown
- **Total VLE Clicks** slider

Real-time predictions from both models with:
- Predicted outcome class
- Confidence scores
- Model agreement indicator

---

## 🔧 Usage Examples

### Train Models with OULAD Data
```bash
python models/train.py
```

Output: Trained models saved to `models/` directory

### Preprocess Data Only
```bash
python utils/preprocess.py
```

Output: Processed dataset statistics

### Use Downloaded OULAD Data
1. Download from: https://analyse.kmi.open.ac.uk/open_dataset
2. Extract CSV files to `data/` folder
3. Run the app: `streamlit run app.py`
4. App automatically loads and processes real OULAD data

### Use Sample Data
```python
from utils.download_oulad import generate_oulad_sample_data

students, assessments, scores, vle = generate_oulad_sample_data()
# Use for testing/development
```

---

## 📊 Expected Results

### With Sample Data
- **Dataset**: ~300 students, ~2000 assessments
- **Classes**: 4 (Pass, Fail, Withdrawn, Distinction)
- **Train/Test**: 80/20 split
- **LR Accuracy**: ~0.75-0.85
- **RF Accuracy**: ~0.85-0.92

### With Real OULAD Data
- **Dataset**: ~32,000+ students across multiple courses
- **Classes**: 4 (Pass, Fail, Withdrawn, Distinction)
- **Much higher accuracy** due to real patterns
- Better insights into actual student success factors

---

## 🌐 Deployment

### Streamlit Cloud (Free & Easy)
1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Connect GitHub account
4. Select repository and `app.py`
5. Deploy!

### Local Server
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Docker
```bash
docker build -t oulad-predictor .
docker run -p 8501:8501 oulad-predictor
```

---

## 📚 Resources

### Dataset
- OULAD Homepage: https://analyse.kmi.open.ac.uk/open_dataset
- Paper: https://doi.org/10.1038/sdata.2017.5

### Libraries
- Streamlit: https://streamlit.io/
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/
- Plotly: https://plotly.com/python/

### ML Resources
- Classification: https://scikit-learn.org/stable/modules/classification.html
- Model Evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Feature Scaling: https://scikit-learn.org/stable/modules/preprocessing.html

---

## 🐛 Troubleshooting

### "CSV files not found"
**Solution**: Download from https://analyse.kmi.open.ac.uk/open_dataset or use sample data

### "ModuleNotFoundError"
**Solution**: `pip install -r requirements.txt`

### "Port 8501 already in use"
**Solution**: `streamlit run app.py --server.port 8502`

### "Models not found"
**Solution**: `python models/train.py`

### "Low accuracy with sample data"
**Solution**: Download real OULAD data for better patterns

---

## Learning Outcomes


✅ Working with real, public ML datasets  
✅ Classification vs Regression tasks  
✅ Feature engineering from raw data  
✅ Categorical encoding strategies  
✅ Model training and evaluation  
✅ Feature importance analysis  
✅ Building interactive ML dashboards  
✅ Handling imbalanced classes  
✅ Production ML workflows  
✅ Data privacy and ethics (using public data)

---

## 📝 Citation

If you use this project or OULAD data in research, please cite:

```bibtex
@article{kuzilek2017open,
  title={Open University Learning Analytics dataset},
  author={Kuzilek, Jakub and Hlosta, Martin and Zdrahal, Zdenek},
  journal={Scientific Data},
  volume={4},
  pages={170005},
  year={2017}
Project by Alireza. Kafi - 2025
}
```

---

## 📄 License

- **Project Code**: APACHE LICENSE
- **OULAD Dataset**: CC BY 4.0
- Attribution required when using OULAD data

---

## 🤝 Contributing :D

Feel free to:
- Add more models
- Improve feature engineering
- Enhance visualizations
- Fix bugs 
- Add documentation

---

**Start with:**
```bash
streamlit run app.py
```


