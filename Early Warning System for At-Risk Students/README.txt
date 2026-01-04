# Early Warning System for At-Risk Students

A machine learning system that predicts student performance and identifies at-risk students based on their study habits, attendance, and participation patterns.

## 📊 Overview

This project analyzes **1,000,000 student records** to predict academic performance and provide early warnings for struggling students. The system achieved **71.75% accuracy** in explaining performance variance.

## 🎯 Features

- Predicts student scores based on behavioral metrics
- Identifies at-risk students early
- Compares multiple ML models (Decision Tree, Random Forest, Linear Regression)
- Comprehensive data visualization and analysis

## 📁 Dataset

**Size**: 1 million students  
**Columns**: 6 (student_id, weekly_self_study_hours, attendance_percentage, class_participation, total_score, grade)

**Grade Distribution**:
- A: 54.86% | B: 25.82% | C: 14.20% | D: 4.50% | F: 0.62%

## 🔍 Key Findings

### Most Important Factor
**Weekly Self-Study Hours** dominates with **99.83% importance**
- Correlation with score: **0.812** (very strong)
- Attendance & participation: <0.1% importance each

### Performance Breakdown by Grade

| Grade | Avg Study Hours | Avg Score |
|-------|-----------------|-----------|
| A     | 19.4 hrs/week   | 96.0      |
| B     | 12.1 hrs/week   | 77.9      |
| C     | 7.7 hrs/week    | 63.6      |
| D     | 3.8 hrs/week    | 49.4      |
| F     | 1.5 hrs/week    | 35.5      |

**At-Risk Threshold**: Students with <8 hours/week self-study

## 🤖 Model Performance

| Model             | R² Score | Avg Error | Time    |
|-------------------|----------|-----------|---------|
| **Random Forest** | **0.7175** | **±6.10** | 44.76s |
| Decision Tree     | 0.7166   | ±6.11     | 3.36s   |
| Linear Regression | 0.6600   | ±7.16     | 0.18s   |

**Best Model**: Random Forest
- Explains 71.75% of score variance
- Average prediction error: ±6.10 points
- Excellent generalization (no overfitting)

### Training Details
- Training: 800,000 students (80%)
- Testing: 200,000 students (20%)
- Training R²: 0.7186 | Test R²: 0.7166 (difference: 0.002)

## 💡 Key Insights

1. **Study hours is the critical factor** - Focus interventions here
2. **Attendance alone doesn't predict success** - Quality > quantity
3. **Students below 8 hrs/week are high-risk** - Need immediate support
4. **Model generalizes well** - Reliable for real-world deployment

## 🚀 Usage

The system helps educators:
- Predict student final scores early in the semester
- Identify struggling students before they fail
- Target interventions based on study habits
- Track and improve prediction accuracy

## 📈 Visualizations

Includes:
- Correlation heatmaps
- Feature importance charts
- Grade distribution plots
- Model comparison graphs
- Prediction accuracy analysis

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- Machine Learning (Decision Tree, Random Forest, Linear Regression)
- Visualization (Matplotlib, Seaborn)

## ⚙️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/early-warning-system.git
cd early-warning-system
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

**Required packages**:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 🚀 How to Run

### 1. Train the Model
```bash
python train_model.py
```
This will:
- Load the student dataset
- Train all three models (Decision Tree, Random Forest, Linear Regression)
- Display performance metrics
- Save the best model

### 2. Make Predictions

**Load the trained model:**
```python
import joblib
import numpy as np

# Load the saved model
model = joblib.load('student_performance_best_model.pkl')
print("✅ Model loaded successfully!")
```

**Predict for a single student:**
```python
# Input: [study_hours, attendance%, participation]
student_data = [[15.5, 85.0, 6.0]]
predicted_score = model.predict(student_data)[0]
print(f"Predicted Score: {predicted_score:.2f}")
```

**Predict for multiple students:**
```python
students = [
    [20.0, 90.0, 7.5],  # Student 1
    [8.0, 70.0, 4.0],   # Student 2  
    [15.0, 85.0, 6.0]   # Student 3
]

predictions = model.predict(students)
for i, score in enumerate(predictions, 1):
    print(f"Student {i}: {score:.2f}")
```

### 3. Visualize Results
```bash
python visualize.py
```
Generates all analysis charts and saves them to the `output/` folder.

## 📂 Project Structure

```
early-warning-system/
│
├── data/
│   └── student_data.csv          # Dataset (1M students)
│
├── models/
│   └── best_model.pkl             # Trained Random Forest model
│
├── src/
│   ├── train_model.py             # Model training script
│   ├── predict.py                 # Prediction script
│   ├── visualize.py               # Visualization script
│   └── utils.py                   # Helper functions
│
├── output/
│   └── visualizations/            # Generated charts
│
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 💻 Usage Example

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('student_performance_best_model.pkl')

# Prepare student data
# Format: [study_hours, attendance%, participation]
student_data = [[12.5, 88.0, 7.0]]

# Make prediction
predicted_score = model.predict(student_data)[0]
print(f"Predicted Score: {predicted_score:.2f}")

# Determine risk level
if predicted_score < 60:
    print("⚠️ HIGH RISK - Immediate intervention needed")
elif predicted_score < 75:
    print("⚡ MODERATE RISK - Monitor closely")
else:
    print("✅ LOW RISK - On track")
```

### For Google Colab Users
If using Google Colab, upload your model file and use the full path:
```python
model = joblib.load('/content/student_performance_best_model.pkl')
```

## 📝 Conclusion

Study time is the strongest predictor of academic success. Early intervention focusing on building effective study habits can significantly improve student outcomes.

---

*Built for educational institutions to support student success through data-driven insights.*
