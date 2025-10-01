# 🎯 StackVote - Ensemble Learning Platform

A minimal Streamlit application for experimenting with ensemble learning methods (Voting, Stacking, Blending) on classification datasets.

## ✨ Features

- **📁 Dataset Upload**: Upload CSV files (max 50MB)
- **👀 Data Preview**: View first rows and class distribution
- **🤖 Model Selection**: Choose from 7 base models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - K-Nearest Neighbors
  - Naive Bayes
- **🔗 Ensemble Methods**: 
  - Voting (Hard) - Majority vote
  - Voting (Soft) - Average probabilities
- **📊 Training & Evaluation**: 
  - Accuracy and F1-Score metrics
  - Visual comparison charts
- **📈 Visualization**: 
  - Bar charts comparing base vs ensemble accuracy
  - Interactive Plotly charts
- **💾 Download**: 
  - Export results as CSV
  - Download trained ensemble model (pickle)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:/Users/projects/StackVote"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload Dataset
1. Navigate to **"Upload Dataset"** page
2. Upload a CSV file (classification task only)
3. Review the data preview and column information
4. Select the target column
5. Click **"Prepare Data for Training"**

### Step 2: Model Selection
1. Navigate to **"Model Selection"** page
2. Select at least 2 base models
3. Choose ensemble method (Hard or Soft Voting)
4. Review configuration summary

### Step 3: Training & Results
1. Navigate to **"Training & Results"** page
2. Click **"Train Models"** button
3. View performance metrics and visualizations
4. Download results (CSV) and trained model (PKL)

## 📊 Example Dataset Format

Your CSV should have features and a target column:

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,class_A
2.3,4.5,6.7,class_B
...
```

## 🎯 MVP Success Criteria

✅ User uploads a dataset and sees a preview  
✅ User selects 2+ base models + Voting ensemble  
✅ App trains models and shows accuracy comparison in charts  
✅ User can download trained ensemble + metrics  

## ❌ Out of Scope (MVP)

- Regression tasks (classification only)
- Deep learning models
- Real-time data streaming
- Cross-validation

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

## 📝 Notes

- Maximum file size: 50MB
- Categorical features are automatically encoded
- Missing values are filled with column means
- Train/test split: 80/20
- All models use `random_state=42` for reproducibility


## 📄 License

MIT License - Feel free to use and modify!

---

Made with ❤️ using Streamlit
