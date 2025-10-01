import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="StackVote - Ensemble Learning Platform",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None

# Header
st.markdown('<div class="main-header">🎯 StackVote</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ensemble Learning Made Simple</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Upload Dataset", "Model Selection", "Training & Results"])

# Available models
AVAILABLE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# ==================== PAGE 1: UPLOAD DATASET ====================
if page == "Upload Dataset":
    st.header("📁 Upload Your Dataset")
    
    st.info("💡 **Requirements:** CSV file, max 50MB, classification task only")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Upload a CSV file with features and target column")
    
    if uploaded_file is not None:
        # Check file size
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        if file_size > 50:
            st.error(f"❌ File size ({file_size:.2f} MB) exceeds 50MB limit!")
        else:
            try:
                # Load data
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                st.success(f"✅ Dataset loaded successfully! ({file_size:.2f} MB)")
                
                # Dataset overview
                st.subheader("📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    st.metric("Missing Values", data.isnull().sum().sum())
                
                # Preview data
                st.subheader("👀 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Column information
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes.values,
                    'Non-Null Count': data.count().values,
                    'Unique Values': [data[col].nunique() for col in data.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
                # Select target column
                st.subheader("🎯 Select Target Column")
                target_col = st.selectbox("Choose the target column for classification", data.columns)
                
                if target_col:
                    # Show class distribution
                    st.subheader("📈 Class Distribution")
                    class_dist = data[target_col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Class Counts:**")
                        st.dataframe(class_dist.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}))
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        class_dist.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Class', fontsize=12)
                        ax.set_ylabel('Count', fontsize=12)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    # Prepare data for training
                    if st.button("🚀 Prepare Data for Training", type="primary"):
                        try:
                            # Separate features and target
                            X = data.drop(columns=[target_col])
                            y = data[target_col]
                            
                            # Handle categorical features
                            categorical_cols = X.select_dtypes(include=['object']).columns
                            if len(categorical_cols) > 0:
                                st.warning(f"⚠️ Found {len(categorical_cols)} categorical columns. Encoding them...")
                                for col in categorical_cols:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                            
                            # Encode target if categorical
                            if y.dtype == 'object':
                                le = LabelEncoder()
                                y = le.fit_transform(y)
                            
                            # Handle missing values
                            if X.isnull().sum().sum() > 0:
                                st.warning("⚠️ Found missing values. Filling with column means...")
                                X = X.fillna(X.mean())
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Store in session state
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            
                            st.success("✅ Data prepared successfully!")
                            st.info(f"📊 Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error preparing data: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")

# ==================== PAGE 2: MODEL SELECTION ====================
elif page == "Model Selection":
    st.header("🤖 Model Selection & Ensemble Configuration")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please upload and prepare your dataset first!")
        st.info("👈 Go to 'Upload Dataset' page to get started")
    else:
        st.success("✅ Dataset is ready for training!")
        
        # Model selection
        st.subheader("🎯 Select Base Models")
        st.info("💡 Choose at least 2 models for ensemble learning")
        
        selected_models = []
        cols = st.columns(2)
        
        for idx, (model_name, model) in enumerate(AVAILABLE_MODELS.items()):
            with cols[idx % 2]:
                if st.checkbox(model_name, key=f"model_{idx}"):
                    selected_models.append(model_name)
        
        if len(selected_models) < 2:
            st.warning("⚠️ Please select at least 2 models for ensemble learning")
        else:
            st.success(f"✅ Selected {len(selected_models)} models: {', '.join(selected_models)}")
            
            # Ensemble method selection
            st.subheader("🔗 Ensemble Method")
            ensemble_method = st.radio(
                "Choose ensemble method",
                ["Voting (Hard)", "Voting (Soft)"],
                help="Hard voting: majority vote | Soft voting: average probabilities"
            )
            
            # Store selections
            st.session_state.selected_models = selected_models
            st.session_state.ensemble_method = ensemble_method
            
            # Display configuration
            st.subheader("📋 Configuration Summary")
            st.write(f"**Base Models:** {', '.join(selected_models)}")
            st.write(f"**Ensemble Method:** {ensemble_method}")
            st.write(f"**Training Samples:** {st.session_state.X_train.shape[0]}")
            st.write(f"**Test Samples:** {st.session_state.X_test.shape[0]}")
            st.write(f"**Features:** {st.session_state.X_train.shape[1]}")
            
            st.info("👉 Go to 'Training & Results' page to train your models!")

# ==================== PAGE 3: TRAINING & RESULTS ====================
elif page == "Training & Results":
    st.header("🚀 Training & Results")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please upload and prepare your dataset first!")
        st.info("👈 Go to 'Upload Dataset' page to get started")
    elif not hasattr(st.session_state, 'selected_models') or len(st.session_state.selected_models) < 2:
        st.warning("⚠️ Please select at least 2 models first!")
        st.info("👈 Go to 'Model Selection' page to choose models")
    else:
        if st.button("🎯 Train Models", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            trained_models = {}
            
            # Train individual models
            total_models = len(st.session_state.selected_models)
            for idx, model_name in enumerate(st.session_state.selected_models):
                status_text.text(f"Training {model_name}...")
                
                model = AVAILABLE_MODELS[model_name]
                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)
                
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                f1 = f1_score(st.session_state.y_test, y_pred, average='weighted')
                
                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'F1-Score': f1
                })
                
                trained_models[model_name] = model
                progress_bar.progress((idx + 1) / (total_models + 1))
            
            # Train ensemble
            status_text.text("Training Ensemble Model...")
            
            voting_type = 'hard' if 'Hard' in st.session_state.ensemble_method else 'soft'
            estimators = [(name, trained_models[name]) for name in st.session_state.selected_models]
            
            ensemble = VotingClassifier(estimators=estimators, voting=voting_type)
            ensemble.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred_ensemble = ensemble.predict(st.session_state.X_test)
            
            accuracy_ensemble = accuracy_score(st.session_state.y_test, y_pred_ensemble)
            f1_ensemble = f1_score(st.session_state.y_test, y_pred_ensemble, average='weighted')
            
            results.append({
                'Model': f'Ensemble ({st.session_state.ensemble_method})',
                'Accuracy': accuracy_ensemble,
                'F1-Score': f1_ensemble
            })
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")
            
            # Store results
            st.session_state.results = pd.DataFrame(results)
            st.session_state.trained_models = trained_models
            st.session_state.ensemble_model = ensemble
            
            st.success("🎉 All models trained successfully!")
            st.balloons()
        
        # Display results
        if st.session_state.results is not None:
            st.subheader("📊 Results")
            
            # Metrics table
            st.dataframe(
                st.session_state.results.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score'], color='lightgreen'),
                use_container_width=True
            )
            
            # Visualization
            st.subheader("📈 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['skyblue'] * (len(st.session_state.results) - 1) + ['coral']
                bars = ax.bar(st.session_state.results['Model'], st.session_state.results['Accuracy'], color=colors, edgecolor='black')
                ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # F1-Score comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['lightblue'] * (len(st.session_state.results) - 1) + ['salmon']
                bars = ax.bar(st.session_state.results['Model'], st.session_state.results['F1-Score'], color=colors, edgecolor='black')
                ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel('F1-Score', fontsize=12)
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Interactive Plotly chart
            st.subheader("🎨 Interactive Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=st.session_state.results['Model'],
                y=st.session_state.results['Accuracy'],
                marker_color='skyblue'
            ))
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=st.session_state.results['Model'],
                y=st.session_state.results['F1-Score'],
                marker_color='coral'
            ))
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download results CSV
                csv = st.session_state.results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="stackvote_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download ensemble model
                if st.session_state.ensemble_model is not None:
                    buffer = io.BytesIO()
                    pickle.dump(st.session_state.ensemble_model, buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Ensemble Model (PKL)",
                        data=buffer,
                        file_name="ensemble_model.pkl",
                        mime="application/octet-stream"
                    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info("""
**StackVote** is a simple ensemble learning platform that helps you:
- Upload datasets
- Select multiple ML models
- Create voting ensembles
- Compare performance
- Download results
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
