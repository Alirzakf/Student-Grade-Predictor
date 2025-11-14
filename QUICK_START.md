# Quick Reference Guide

## 🚀 First Time Setup (5 minutes)

### Windows
```bash
cd "Student Grade Predictor"
pip install -r requirements.txt
streamlit run app.py
```

Or simply double-click: `run.bat`

### macOS/Linux
```bash
cd "Student Grade Predictor"
bash run.sh
```

---

## 📚 Common Commands

### Train Models Only
```bash
python models/train.py
```

### Launch Dashboard
```bash
streamlit run app.py
```

### Test Preprocessing
```bash
python utils/preprocess.py
```

### Generate New Synthetic Data
```bash
python -c "from utils.preprocess import generate_synthetic_data; df = generate_synthetic_data(300); df.to_csv('data/students.csv', index=False)"
```

### Use Custom Data
1. Place your CSV in `data/students.csv`
2. Ensure columns: `study_hours, attendance, previous_grade, sleep_hours, internet_usage, final_grade`
3. Delete `models/*.pkl` files
4. Run: `streamlit run app.py`

---

## 🎯 Dashboard Navigation

### Overview Page
- **Dataset Statistics**: Count, features, average grade
- **Data Preview**: First 20 rows
- **Statistics Table**: Mean, std, min, max, quartiles
- **Correlation Heatmap**: Feature relationships
- **Feature Importance**: Random Forest importance scores
- **Coefficients**: Linear Regression impact

### What-If Prediction Page
- **Study Hours** (0-12): Average daily study time
- **Attendance** (0-100%): Class attendance percentage
- **Previous Grade** (0-100): Last semester grade
- **Sleep Hours** (0-12): Average nightly sleep
- **Internet Usage** (0-12): Non-study internet hours

**Predictions**: Real-time from both models
**Insights**: Personalized recommendations

---

## 📊 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `models/train.py` | Training script |
| `utils/preprocess.py` | Data preprocessing |
| `utils/visualize.py` | Plotting functions |
| `data/students.csv` | Dataset (auto-generated) |
| `requirements.txt` | Dependencies |

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Clear Cache
```bash
rm -rf .streamlit
# or on Windows: rmdir /s .streamlit
```

### Reinstall Dependencies
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Models Not Found
```bash
python models/train.py
```

---

## 📈 Expected Results

**With Synthetic Data:**
- Dataset: 200 student records
- Train/Test Split: 80/20 (160/40)
- Linear Regression R²: ~0.85
- Random Forest R²: ~0.92
- Test RMSE: ~6 grade points

---

### Local Server
```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 📝 Project Structure

```
Student Grade Predictor/
├── app.py                  # Main app
├── requirements.txt        # Dependencies
├── README.md              # Full docs
├── SETUP.md               # Installation guide
├── run.bat/run.sh         # Quick start scripts
├── models/
│   ├── train.py          # Training code
│   └── (pkl files)        # Trained models
├── utils/
│   ├── preprocess.py     # Data processing
│   └── visualize.py      # Plotting
└── data/
    └── students.csv      # Dataset
```

---

## 💡 Tips & Tricks

### Speed Up Training
```python
# Reduce dataset size in preprocess.py
generate_synthetic_data(n_samples=50)
```

### Change Plot Style
Edit colors in `utils/visualize.py`:
```python
marker=dict(color='steelblue')  # Change color
```

### Add New Features
1. Edit `generate_synthetic_data()` in `preprocess.py`
2. Retrain models
3. Update app.py sliders

### Export Predictions
```python
from models.train import StudentGradePredictor
predictor = StudentGradePredictor()
results, data = predictor.train_models()
# results contains all predictions and metrics
```

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Pandas Docs](https://pandas.pydata.org/)
- [Plotly Docs](https://plotly.com/python/)

---

## ❓ FAQs

**Q: Where does the data come from?**
A: Automatically generated synthetically if not provided.

**Q: Can I use my own data?**
A: Yes! Place CSV in `data/students.csv` with required columns.

**Q: How do I deploy this?**
A: See README.md deployment section.

**Q: Can I add more models?**
A: Yes! Edit `models/train.py` and `app.py`.

**Q: Is it production-ready?**
A: Yes! Modular, documented, and deployable.

---

**Need help? Check README.md or SETUP.md**
