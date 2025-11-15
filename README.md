# Student Grade Predictor

A machine learning application using a real UK public dataset to predict student outcomes.

## 🎯 Project Overview

This project uses the **Open University Learning Analytics Dataset (OULAD)** - a real UK public dataset - to predict student final results (Pass, Fail, Withdrawn, Distinction) based on learning analytics and behavioral data.

**Dataset**: Open University Learning Analytics Dataset (OULAD)  
**Source**: https://analyse.kmi.open.ac.uk/open_dataset  
**License**: CC BY 4.0

> 📖 **For detailed OULAD information**: See `README_OULAD.md` for dataset structure, features, and how to use real OULAD data.

### Features
- **Real Public Data**: Uses actual OULAD dataset from Open University UK
- **Classification Models**: Logistic Regression and Random Forest Classifier
- **Feature Analysis**: Feature importance and coefficient visualization
- **Interactive Dashboard**: Streamlit app with real-time predictions
- **Production Ready**: Professional-grade code and documentation

## 📋 Project Structure

```
project/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── README_OULAD.md                # OULAD dataset guide
├── data/
│   ├── students.csv               # OULAD students data
│   ├── studentVle.csv             # VLE interactions
│   ├── assessments.csv            # Assessment data
│   └── studentAssessment.csv      # Assessment scores
├── models/
│   ├── train.py                   # Training script
│   ├── logistic_regression.pkl    # Trained classifier
│   ├── random_forest.pkl          # Trained classifier
│   ├── scaler.pkl                 # Feature scaler
│   ├── encoders.pkl               # Categorical encoders
│   ├── label_encoder.pkl          # Target encoder
│   └── feature_names.pkl          # Feature names
└── utils/
    ├── preprocess.py              # OULAD preprocessing
    ├── download_oulad.py          # OULAD data loader
    └── visualize.py               # Visualization functions
```

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "Student Grade Predictor"
```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Option 2: Train Models Only

```bash
python models/train.py
```

This will:
- Generate synthetic data if `data/students.csv` doesn't exist
- Train Linear Regression and Random Forest models
- Save models to the `models/` directory
- Display model comparison and evaluation metrics

#### Option 3: Test Preprocessing

```bash
python utils/preprocess.py
```

## 📊 Features

### 1. Data Preprocessing (`utils/preprocess.py`)

- **OULAD Data Loading**: Loads real OULAD CSV files or generates realistic sample data
- **Categorical Encoding**: LabelEncoding for demographic features (age_band, gender, disability, region)
- **Numeric Feature Scaling**: StandardScaler for academic and engagement features
- **Missing Value Handling**: Mean imputation for numeric features
- **Stratified Train/Test Split**: 80/20 split preserving class distribution

### 2. Model Training (`models/train.py`)

#### Logistic Regression Classifier
- Multi-class classification model
- Shows feature coefficients (feature importance)
- Provides class probabilities
- Suitable for interpretable results

#### Random Forest Classifier
- Ensemble method with 100 trees
- Captures non-linear relationships

## GitHub readiness and large artifacts

Please avoid committing large binary artifacts (trained models, raw datasets, weights) to this repository. Recommended patterns:

- Add model artifacts to `.gitignore` (already included). The `models/` folder is ignored by default for `*.pkl` files.
- Use Git LFS for model files if you want to store them in the repo:

```bash
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
```

- Store raw OULAD CSV files outside Git or provide a small sample and document how to download the full dataset.

Where to put data locally:
- Place real OULAD CSVs under `data/` if you want local testing. Note the `.gitignore` currently ignores `data/*.csv` to avoid accidental commits — change this if you intend to commit small sample files.

CI notes:
- A minimal GitHub Actions workflow is included at `.github/workflows/ci.yml` that installs dependencies and runs a quick import smoke test.


#### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Classification Report**: Per-class performance metrics

### 3. Visualizations (`utils/visualize.py`)

- **Correlation Heatmap**: Shows relationships between features
- **Feature Importance**: Bar chart of Random Forest feature importance
- **Logistic Regression Coefficients**: Shows impact of each feature
- **Confusion Matrix**: Classification accuracy visualization
- **ROC Curves**: Model performance across classification thresholds

### 4. Streamlit Dashboard (`app.py`)

#### Overview Page
- Dataset summary statistics (students, features, class distribution)
- Raw OULAD data preview
- Correlation heatmap
- Feature importance visualization (Random Forest)
- Logistic regression coefficients

#### What-If Prediction Page
- **Age Band** dropdown (≤35, 35-55, >55)
- **Gender** dropdown (M, F)
- **Disability** dropdown (Y, N)
- **Studied Credits** slider (30-120)
- **Previous Attempts** slider (0-5)
- **Average Assessment Score** slider (0-100)
- **Days Since Registration** slider (0-300)
- **Total VLE Clicks** slider (0-500)
- **Region** dropdown (UK regions)
- Real-time predictions with class probabilities
- Model agreement indicator

## 🔧 Configuration

### Using Real OULAD Data

Download from: https://analyse.kmi.open.ac.uk/open_dataset

Extract these files to the `data/` folder:
- `students.csv`
- `studentVle.csv`
- `assessments.csv`
- `studentAssessment.csv`

The app will automatically load and process them.

### Using Sample Data

If OULAD files are not found, the app automatically generates realistic OULAD-like sample data for testing.

### Modifying Model Parameters

Edit `models/train.py`:
```python
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    random_state=42,       # Reproducibility
    n_jobs=-1,            # Use all CPU cores
    max_depth=15           # Maximum tree depth
)
```

### Changing Train/Test Split

```python
preprocess_oulad_data(test_size=0.25)  # 75% train, 25% test
```

## 📈 Expected Performance

With sample OULAD data:
- Logistic Regression Accuracy: ~0.75-0.85
- Random Forest Classifier Accuracy: ~0.85-0.92

With real OULAD data:
- Both models achieve higher accuracy due to real patterns
- Better insights into actual student success factors

Performance will vary based on your actual dataset.

## 🌐 Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- GitHub repository with this code

### Steps

1. **Push code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/student-grade-predictor.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch, and main file (`app.py`)
   - Click "Deploy"

3. **Configure Secrets (if using external data):**
   - Go to app settings
   - Add secrets in `.streamlit/secrets.toml` format

### Notes for Production
- Place sensitive data in `.streamlit/secrets.toml`
- Use `st.secrets` to access secrets in your app
- Monitor app performance and usage

## 📦 Dependencies

All required packages are listed in `requirements.txt`:

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.0.0
```

## 🧪 Testing

### Test OULAD Data Loading
```python
from utils.download_oulad import load_or_generate_oulad
students, assessments, vle = load_or_generate_oulad()
print(f"Loaded {len(students)} students")
```

### Test Data Preprocessing
```python
from utils.preprocess import preprocess_oulad_data
X_train, X_test, y_train, y_test, encoders, label_encoder, y_labels = preprocess_oulad_data()
print(f"X_train shape: {X_train.shape}")
print(f"Classes: {label_encoder.classes_}")
```

### Test Model Training
```python
from models.train import StudentGradePredictor
predictor = StudentGradePredictor()
results, data = predictor.train_models()
print(results)
```

### Test Streamlit App Locally
```bash
streamlit run app.py
```

## 🐛 Troubleshooting

### Issue: "CSV files not found"
**Solution**: Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset or use sample data

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Models not found when running app
**Solution**: Train models first
```bash
python models/train.py
```

### Issue: "Port 8501 already in use"
**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Streamlit app won't start
**Solution**: Check for errors with debug logging
```bash
streamlit run app.py --logger.level=debug
```

### Issue: Memory error with large datasets
**Solution**: Use real OULAD data selectively or adjust preprocessing

## 📝 Example Usage

### Using the trained models in your own code

```python
import pickle
import numpy as np
from utils.preprocess import preprocess_oulad_data

# Load models and encoders
with open('models/logistic_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare categorical input (example)
categorical_data = {'age_band': '35-55', 'gender': 'M', 'disability': 'N', 'region': 'East'}
numeric_data = np.array([[60, 2, 150, 65, 200]])  # [credits, attempts, clicks, score, days]

# Encode categorical features
for col, value in categorical_data.items():
    categorical_data[col] = encoders[col].transform([value])[0]

# Scale numeric features
numeric_scaled = scaler.transform(numeric_data)

# Combine and predict
student_features = np.concatenate([list(categorical_data.values()), numeric_scaled[0]])
prediction = lr_model.predict(student_features.reshape(1, -1))
print(f"Predicted outcome: {prediction[0]}")
```

## 📚 Understanding the Models

### Logistic Regression Classifier
- **Pros**: Fast, interpretable, shows feature importance
- **Cons**: Assumes linear decision boundaries
- **Best For**: Understanding which features impact outcomes

### Random Forest Classifier
- **Pros**: Handles non-linear patterns, high accuracy
- **Cons**: Less interpretable, requires more data
- **Best For**: Maximum prediction accuracy

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)

## 📄 License

- check license file for more information