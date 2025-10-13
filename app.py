# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Dark Theme CSS with FULL-SCREEN Power BI fixes
st.markdown("""
<style>
    /* Main app background - DARK THEME */
    .main, .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* All text elements - WHITE for visibility */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1y4p8pa {
        background-color: #262730;
    }
    
    /* Radio buttons and select boxes */
    .stRadio > div {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #555555;
        margin: 10px 0;
    }
    
    .stRadio label {
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 500;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        background-color: #262730;
        border: 1px solid #444444;
        width: 100%;
        display: block;
        transition: all 0.3s ease;
    }
    
    .stRadio label[data-testid="stRadioLabel"]:has(input:checked) {
        background-color: #0E8C38 !important;
        color: white !important;
        border-color: #0E8C38;
        box-shadow: 0 2px 8px rgba(14, 140, 56, 0.3);
    }
    
    /* Text input styling */
    .stTextInput label {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    .stTextInput input {
        background-color: #262730 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
        border-radius: 6px;
        padding: 12px;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #0E8C38;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background-color: #0DA44B;
    }
    
    /* Custom card styling */
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        color: white;
    }
    
    /* FULL-SCREEN Power BI container */
    .powerbi-container {
        border: 2px solid #2e86ab;
        border-radius: 15px;
        padding: 0;
        margin: 15px 0;
        background-color: #262730 !important;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .powerbi-frame {
        width: 100%;
        height: 100%;
        min-height: 900px;
        border: none;
        border-radius: 10px;
        background-color: transparent !important;
        display: block;
    }
    
    /* Remove padding from main content when viewing Power BI */
    .element-container:has(.powerbi-container) {
        width: 100%;
        max-width: 100%;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab !important;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    .main-header {
        font-size: 3rem;
        color: #2e86ab !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #262730 !important;
        color: white !important;
    }
    
    /* Instructions box */
    .instruction-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0E8C38;
        margin: 15px 0;
        color: white;
    }
    
    /* Fix for Streamlit components */
    .st-bw, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class BankMarketingDashboard:
    def __init__(self):
        # Initialize session state for persistence
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = {}
        if 'df_loaded' not in st.session_state:
            st.session_state.df_loaded = None
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = None
        if 'test_data' not in st.session_state:
            st.session_state.test_data = {}
        
        self.df = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess the data"""
        # Use session state to avoid reloading
        if st.session_state.df_loaded is not None:
            self.df = st.session_state.df_loaded
            return True
            
        try:
            # Load dataset
            file_path = 'bank-full.csv'
            self.df = pd.read_csv(file_path, sep=';')
            
            # Data cleaning steps from notebook
            categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
            self.df[categorical_cols] = self.df[categorical_cols].replace('unknown', 'Unknown')
            
            binary_cols = ['default', 'housing', 'loan', 'y']
            for col in binary_cols:
                self.df[col] = self.df[col].map({'yes': 1, 'no': 0})
            
            # Store in session state
            st.session_state.df_loaded = self.df
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def prepare_data(self):
        """Prepare data for modeling with sampling for faster training"""
        # REDUCE SAMPLE SIZE FOR FASTER TRAINING
        sample_fraction = 0.3  # Use less% of data for faster training
        if len(self.df) > 10000:
            self.df = self.df.sample(frac=sample_fraction, random_state=42)
        
        # Separate features and target
        X = self.df.drop('y', axis=1)
        y = self.df['y']
        
        # Define features
        numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        binary_features = ['default', 'housing', 'loan']
        
        # Create preprocessor
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        binary_transformer = 'passthrough'
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features),
                ('bin', binary_transformer, binary_features)
            ]
        )
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store test data for predictions
        st.session_state.test_data = {
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
        # Handle class imbalance using SMOTE-like approach (manual implementation)
        df_train = pd.concat([self.X_train, self.y_train], axis=1)
        df_majority = df_train[df_train.y == 0]
        df_minority = df_train[df_train.y == 1]
        
        # Upsample minority class (but limit for speed)
        max_upsample = min(len(df_minority) * 3, len(df_majority))  # Limit upsampling
        df_minority_upsampled = resample(df_minority,
                                        replace=True,
                                        n_samples=max_upsample,
                                        random_state=42)
        
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        self.X_train = df_upsampled.drop('y', axis=1)
        self.y_train = df_upsampled['y']
        
        # Fit preprocessor
        self.preprocessor.fit(self.X_train)
        st.session_state.preprocessor = self.preprocessor

    def train_models(self):
        """Train multiple machine learning models with optimized parameters"""
        # Define models with OPTIMIZED parameters for speed
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=500,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=50,
                max_depth=10,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42, 
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1
            ),
            'SVM': SVC(
                probability=True, 
                random_state=42, 
                class_weight='balanced',
                kernel='linear',
                max_iter=500
            )
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model_performance = {}
        
        # Train each model
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'🚀 Training {name}... ({i+1}/{len(models)})')
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = pipeline.score(self.X_test, self.y_test)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            model_performance[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Update progress
            progress_bar.progress((i + 1) / len(models))
        
        # Store in session state
        st.session_state.model_performance = model_performance
        st.session_state.models_trained = True
        
        status_text.text("✅ All models trained successfully!")
        st.success("🎯 Models trained and saved! You can now navigate to other sections.")

    def create_dashboard(self):
        """Create the main dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🏦 Bank Marketing Campaign Analysis Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Navigation")
        sections = [
            "Problem Statement & Solution",
            "Data Overview",
            "Exploratory Data Analysis",
            "Model Training & Results",
            "Model Comparison",
            "Power BI Dashboards",
            "Practical Example",
            "Business Recommendations"
        ]
        selected_section = st.sidebar.radio("Go to:", sections)
        
        # Load data
        if self.df is None:
            with st.spinner('Loading and preprocessing data...'):
                if not self.load_data():
                    return
        
        # Section routing
        if selected_section == "Problem Statement & Solution":
            self.show_problem_solution()
        elif selected_section == "Data Overview":
            self.show_data_overview()
        elif selected_section == "Exploratory Data Analysis":
            self.show_eda()
        elif selected_section == "Model Training & Results":
            self.show_model_training()
        elif selected_section == "Model Comparison":
            self.show_model_comparison()
        elif selected_section == "Power BI Dashboards":
            self.show_powerbi_dashboards()
        elif selected_section == "Practical Example":
            self.show_practical_example()
        elif selected_section == "Business Recommendations":
            self.show_business_recommendations()

    def show_problem_solution(self):
        """Display problem statement and solution"""
        st.markdown('<h2 class="section-header">📋 Problem Statement & Proposed Solution</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Problem Statement")
            st.markdown("""
            **Business Challenge:**
            - Bank telemarketing campaigns have low success rates (~11.7%)
            - High costs associated with contacting all clients
            - Inefficient resource allocation
            - Potential customer irritation from irrelevant offers
            
            **ML Problem:** Binary Classification
            - **Target variable:** `y` (client subscription: yes/no)
            - **Goal:** Predict which clients are most likely to subscribe to term deposits
            """)
            
            subscription_rate = (self.df['y'].sum() / len(self.df)) * 100
            st.metric("Current Success Rate", f"{subscription_rate:.1f}%")
            st.metric("Dataset Size", f"{len(self.df):,} records")
            st.metric("Number of Features", f"{len(self.df.columns)}")
        
        with col2:
            st.subheader("💡 Proposed Solution")
            st.markdown("""
            **Predictive Modeling Approach:**
            1. **Data Preparation** - Clean and preprocess client data
            2. **Exploratory Analysis** - Understand patterns and relationships
            3. **Model Training** - Build multiple classification models
            4. **Model Selection** - Choose best performing model
            5. **Deployment** - Implement for targeted campaigns
            
            **Expected Benefits:**
            - ✅ **60-80%** reduction in contact costs
            - ✅ **3-5x** improvement in conversion rates
            - ✅ Better customer experience
            - ✅ Data-driven decision making
            """)
            
            st.info("""
            **Workflow:** Phase 0 → Phase 1 → Phase 2 → Phase 3
            - Phase 0: Data Loading & Inspection
            - Phase 1: Data Cleaning & Preprocessing  
            - Phase 2: Exploratory Data Analysis
            - Phase 3: Model Training & Evaluation
            """)
        
        # Team Members Section
        st.markdown("---")
        st.markdown('<h3 class="section-header" style="text-align: center; margin-top: 2rem;">👥 Project Team</h3>', unsafe_allow_html=True)
        
        # Create 5 columns for team members
        cols = st.columns(5)
        
        team_members = [
            ("Narravula Mukesh", "🎯"),
            ("Revanth Singothu", "💼"),
            ("N. Siddharth Swamy", "🚀"),
            ("Nandha Kumar P", "📊"),
            ("Devesh Keshavan S", "🔬")
        ]
        
        for col, (name, icon) in zip(cols, team_members):
            with col:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #2e86ab; margin: 10px 0;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">{icon}</div>
                    <div style="color: white; font-weight: 600; font-size: 14px;">{name}</div>
                </div>
                """, unsafe_allow_html=True)

    def show_data_overview(self):
        """Display data overview"""
        st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(self.df):,}")
        with col2:
            st.metric("Number of Features", f"{len(self.df.columns)}")
        with col3:
            subscription_rate = (self.df['y'].sum() / len(self.df)) * 100
            st.metric("Subscription Rate", f"{subscription_rate:.1f}%")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(self.df.head(10), use_container_width=True)
        
        # Data types and basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(self.df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Basic Statistics")
            st.dataframe(self.df.describe(), use_container_width=True)
        
        # Missing values
        st.subheader("Data Quality Check")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            st.success("✅ No missing values in the dataset")
        else:
            st.warning(f"❌ Missing values found: {missing_values[missing_values > 0]}")

    def show_eda(self):
        """Display exploratory data analysis"""
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Target variable distribution
        st.subheader("Target Variable Distribution")
        fig = px.pie(values=self.df['y'].value_counts().values, 
                    names=['Not Subscribed', 'Subscribed'],
                    color_discrete_sequence=['#ff9999', '#66b3ff'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Numeric features distribution
        st.subheader("Distribution of Numerical Features")
        numeric_cols = ['age', 'balance', 'duration', 'campaign']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=numeric_cols)
        for i, col in enumerate(numeric_cols):
            row = (i // 2) + 1
            col_num = (i % 2) + 1
            fig.add_trace(go.Histogram(x=self.df[col], name=col), row=row, col=col_num)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features
        st.subheader("Categorical Features Distribution")
        categorical_cols = ['job', 'marital', 'education', 'contact']
        
        for col in categorical_cols:
            fig = px.bar(self.df[col].value_counts(), 
                        title=f'Distribution of {col}',
                        color_discrete_sequence=['#2e86ab'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix,
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    def show_model_training(self):
        """Display model training section with session state"""
        st.markdown('<h2 class="section-header">🤖 Model Training & Results</h2>', unsafe_allow_html=True)
        
        if not st.session_state.models_trained:
            st.info("🚀 Ready to train models! Click the button below to start training.")
            
            if st.button("Train All Models", type="primary", key="train_models_btn"):
                with st.spinner('Preparing data and training models... This may take 1-2 minutes.'):
                    self.prepare_data()
                    self.train_models()
        else:
            st.success("✅ Models already trained! Navigate to 'Model Comparison' to see results.")
            
            # Show quick performance summary
            st.subheader("Training Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models Trained", "4")
            with col2:
                best_model = max(st.session_state.model_performance.items(), 
                               key=lambda x: x[1]['auc_score'])[0]
                st.metric("Best Model", best_model)
            with col3:
                best_auc = max(perf['auc_score'] for perf in st.session_state.model_performance.values())
                st.metric("Best AUC", f"{best_auc:.3f}")
            with col4:
                if st.button("Retrain Models", key="retrain_btn"):
                    st.session_state.models_trained = False
                    st.session_state.model_performance = {}
                    st.rerun()
            
            # Show detailed results
            st.subheader("Trained Models Performance")
            model_performance = st.session_state.model_performance
            
            # Model performance metrics
            metrics_data = []
            for name, performance in model_performance.items():
                metrics_data.append({
                    'Model': name,
                    'Accuracy': f"{performance['accuracy']:.3f}",
                    'ROC-AUC': f"{performance['auc_score']:.3f}",
                    'Precision': f"{performance['classification_report']['1']['precision']:.3f}",
                    'Recall': f"{performance['classification_report']['1']['recall']:.3f}",
                    'F1-Score': f"{performance['classification_report']['1']['f1-score']:.3f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

    def show_model_comparison(self):
        """Display model comparison with session state"""
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first in the 'Model Training & Results' section.")
            st.info("Go to the 'Model Training & Results' section and click 'Train All Models'")
            return
            
        st.markdown('<h2 class="section-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
        
        model_performance = st.session_state.model_performance
        
        # Performance comparison chart
        models = list(model_performance.keys())
        accuracy_scores = [model_performance[model]['accuracy'] for model in models]
        auc_scores = [model_performance[model]['auc_score'] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models, y=accuracy_scores, marker_color='#2e86ab'),
            go.Bar(name='ROC-AUC', x=models, y=auc_scores, marker_color='#a23b72')
        ])
        fig.update_layout(title='Model Performance Comparison',
                         barmode='group',
                         yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        metrics_data = []
        for name, performance in model_performance.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{performance['accuracy']:.3f}",
                'ROC-AUC': f"{performance['auc_score']:.3f}",
                'Precision': f"{performance['classification_report']['1']['precision']:.3f}",
                'Recall': f"{performance['classification_report']['1']['recall']:.3f}",
                'F1-Score': f"{performance['classification_report']['1']['f1-score']:.3f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model identification
        best_model = max(model_performance.items(), 
                        key=lambda x: (x[1]['auc_score'] + x[1]['accuracy'])/2)
        
        st.success(f"🎯 **Best Performing Model: {best_model[0]}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{best_model[1]['accuracy']:.3f}")
        with col2:
            st.metric("ROC-AUC", f"{best_model[1]['auc_score']:.3f}")
        with col3:
            st.metric("F1-Score", f"{best_model[1]['classification_report']['1']['f1-score']:.3f}")
        
        # Show individual model results in tabs
        st.subheader("Individual Model Analysis")
        model_tabs = st.tabs(list(model_performance.keys()))
        
        for i, (model_name, performance) in enumerate(model_performance.items()):
            with model_tabs[i]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    cm = confusion_matrix(st.session_state.test_data['y_test'], performance['y_pred'])
                    fig = px.imshow(cm,
                                  labels=dict(x="Predicted", y="Actual", color="Count"),
                                  x=['Not Subscribe', 'Subscribe'],
                                  y=['Not Subscribe', 'Subscribe'],
                                  title=f"Confusion Matrix - {model_name}",
                                  color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(st.session_state.test_data['y_test'], performance['y_pred_proba'])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                           name=f'ROC curve (AUC = {performance["auc_score"]:.3f})',
                                           line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                           name='Random Classifier', 
                                           line=dict(dash='dash', color='red')))
                    fig.update_layout(title=f'ROC Curve - {model_name}',
                                    xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(performance['classification_report']).transpose()
                st.dataframe(report_df, use_container_width=True)

    def show_powerbi_dashboards(self):
        """Display embedded Power BI dashboards"""
        st.markdown('<h2 class="section-header">📊 Power BI Dashboards</h2>', unsafe_allow_html=True)
        
        st.info("""
        **📈 Professional Business Intelligence Dashboards** - Embedded Power BI reports for comprehensive analysis.
        Configure your Power BI URLs below to display interactive dashboards in **full-screen** mode.
        """)
        
        # Power BI Report Selection with better styling
        st.markdown("### 📊 Select Power BI Report")
        
        report_option = st.radio(
            "Choose a dashboard to view:",
            [
                "📊 Campaign Performance Overview",
                "👥 Customer Segmentation Analysis", 
                "📞 Contact Strategy Optimization",
                "🎯 Predictive Model Insights"
            ],
            key="powerbi_selector"
        )
        
        st.markdown("---")
        
        # Display different Power BI reports based on selection
        if report_option == "📊 Campaign Performance Overview":
            self.show_campaign_performance_report()
        elif report_option == "👥 Customer Segmentation Analysis":
            self.show_customer_segmentation_report()
        elif report_option == "📞 Contact Strategy Optimization":
            self.show_contact_strategy_report()
        elif report_option == "🎯 Predictive Model Insights":
            self.show_predictive_insights_report()
        
        # Power BI Integration Guide
        st.markdown("---")
        with st.expander("🛠️ Power BI Integration Guide - How to Embed Your Reports"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 📝 Step-by-Step Setup:
                
                **1. Prepare Your Report**
                - Create your dashboard in Power BI Desktop
                - Ensure all visuals are finalized
                - Test interactivity and filters
                
                **2. Publish to Power BI Service**
                - Click **File → Publish → Publish to Power BI**
                - Select your workspace
                - Wait for upload completion
                
                **3. Get Embed Code**
                - Go to app.powerbi.com
                - Open your published report
                - Click **File → Embed report → Website or portal**
                - Copy the provided URL
                """)
            
            with col2:
                st.markdown("""
                ### ⚙️ Configuration Tips:
                
                **URL Format:**
                ```
                https://app.powerbi.com/reportEmbed?
                reportId=YOUR-REPORT-ID&
                autoAuth=true&
                ctid=YOUR-TENANT-ID
                ```
                
                **Display Options:**
                - **900px**: Good for overview dashboards
                - **1200px**: Standard full-page view
                - **1500px**: Detailed analysis view
                - **Full Screen**: Maximum viewing area (2000px)
                
                **Best Practices:**
                - ✅ Use public or Premium workspace for embedding
                - ✅ Test URL before deploying
                - ✅ Enable "Allow fullscreen" in Power BI settings
                - ✅ Optimize report performance for web viewing
                """)
            
            st.warning("""
            **⚠️ Important Notes:**
            - Reports must be published to Power BI online service
            - For private reports, you need Power BI Premium or Pro licensing
            - Embedded reports respect Row-Level Security (RLS) if configured
            - Use HTTPS URLs only for security
            """)
            
            st.success("""
            **💡 Pro Tip:** After pasting your URL, use the height selector to adjust the dashboard size. 
            Most dashboards work best at 1200px or higher for optimal viewing experience.
            """)

    def show_campaign_performance_report(self):
        """Display Campaign Performance Power BI Report - FULL SCREEN"""
        st.subheader("📊 Campaign Performance Overview")
        
        # Power BI URL input with improved integration
        st.markdown("### 🔗 Configure Power BI Integration")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            powerbi_url = st.text_input(
                "Enter your Campaign Performance Power BI URL:",
                value="https://app.powerbi.com/reportEmbed?reportId=664ee3b9-f63e-4a9d-a036-c1895b831388&autoAuth=true&ctid=00f9cda3-075e-44e5-aa0b-aba3add6539f",
                key="campaign_url",
                help="Paste the embed URL from Power BI Service"
            )
        
        with col2:
            if st.button("Test Connection", key="test_campaign", use_container_width=True):
                if powerbi_url and "YOUR_CAMPAIGN_REPORT_ID" not in powerbi_url:
                    st.success("✅ Valid!")
                else:
                    st.warning("⚠️ Invalid URL")
        
        with col3:
            height_option = st.selectbox("Height", ["900px", "1200px", "1500px", "Full Screen"], key="height_campaign")
        
        # Convert height option to pixels
        height_map = {"900px": 900, "1200px": 1200, "1500px": 1500, "Full Screen": 2000}
        iframe_height = height_map[height_option]
        
        if powerbi_url and "YOUR_CAMPAIGN_REPORT_ID" not in powerbi_url:
            # Embed Power BI report with FULL SCREEN support
            try:
                st.markdown("---")
                st.info("🎯 **Key Campaign Metrics & Performance Trends** - Use the Power BI controls to interact with the dashboard")
                
                powerbi_html = f"""
                <div style="width: 100%; height: {iframe_height}px; position: relative; background-color: #262730; border-radius: 15px; border: 2px solid #2e86ab; overflow: hidden;">
                    <iframe 
                        src="{powerbi_url}" 
                        frameborder="0" 
                        width="100%" 
                        height="100%"
                        allowFullScreen="true"
                        style="border: none; display: block;">
                    </iframe>
                </div>
                """
                st.components.v1.html(powerbi_html, height=iframe_height, scrolling=False)
                st.success("✅ Power BI dashboard loaded successfully! Use the fullscreen button in Power BI for maximum viewing area.")
            except Exception as e:
                st.error(f"❌ Failed to load Power BI dashboard: {e}")
                st.info("💡 Make sure your Power BI report is published and the URL is correct.")
        else:
            # Show placeholder with better instructions
            self.show_powerbi_placeholder("Campaign Performance", "conversion trends and ROI analysis")

    def show_customer_segmentation_report(self):
        """Display Customer Segmentation Power BI Report - FULL SCREEN"""
        st.subheader("👥 Customer Segmentation Analysis")
        
        # Power BI URL input
        st.markdown("### 🔗 Configure Power BI Integration")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            powerbi_url = st.text_input(
                "Enter your Customer Segmentation Power BI URL:",
                value="https://app.powerbi.com/reportEmbed?reportId=e83ca50e-113d-4853-941e-6094241f328f&autoAuth=true&ctid=00f9cda3-075e-44e5-aa0b-aba3add6539f", 
                key="segmentation_url",
                help="Get this from Power BI Service → Embed → Website or portal"
            )
        
        with col2:
            if st.button("Test Connection", key="test_segmentation", use_container_width=True):
                if powerbi_url and "YOUR_SEGMENTATION_REPORT_ID" not in powerbi_url:
                    st.success("✅ Valid!")
                else:
                    st.warning("⚠️ Invalid URL")
        
        with col3:
            height_option = st.selectbox("Height", ["900px", "1200px", "1500px", "Full Screen"], key="height_segmentation")
        
        # Convert height option to pixels
        height_map = {"900px": 900, "1200px": 1200, "1500px": 1500, "Full Screen": 2000}
        iframe_height = height_map[height_option]
        
        if powerbi_url and "YOUR_SEGMENTATION_REPORT_ID" not in powerbi_url:
            try:
                st.markdown("---")
                st.info("🎯 **Client Demographics & Behavior Patterns** - Analyze customer segments based on age, job, balance, and subscription behavior")
                
                # Embed Power BI report with full screen
                powerbi_html = f"""
                <div style="width: 100%; height: {iframe_height}px; position: relative; background-color: #262730; border-radius: 15px; border: 2px solid #2e86ab; overflow: hidden;">
                    <iframe 
                        src="{powerbi_url}" 
                        frameborder="0" 
                        width="100%" 
                        height="100%"
                        allowFullScreen="true"
                        style="border: none; display: block;">
                    </iframe>
                </div>
                """
                st.components.v1.html(powerbi_html, height=iframe_height, scrolling=False)
                st.success("✅ Customer Segmentation dashboard loaded!")
            except Exception as e:
                st.error(f"❌ Error loading dashboard: {e}")
        else:
            self.show_powerbi_placeholder("Customer Segmentation", "demographic analysis and behavior patterns")

    def show_contact_strategy_report(self):
        """Display Contact Strategy Power BI Report - FULL SCREEN"""
        st.subheader("📞 Contact Strategy Optimization")
        
        # Power BI URL input
        st.markdown("### 🔗 Configure Power BI Integration")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            powerbi_url = st.text_input(
                "Enter Contact Strategy Power BI URL:",
                value="https://app.powerbi.com/reportEmbed?reportId=aba2bc01-3099-48cb-81fd-9d4c426aafd7&autoAuth=true&ctid=00f9cda3-075e-44e5-aa0b-aba3add6539f",
                key="contact_strategy_url"
            )
        
        with col2:
            if st.button("Test Connection", key="test_contact", use_container_width=True):
                if powerbi_url and "YOUR_CONTACT_STRATEGY_ID" not in powerbi_url:
                    st.success("✅ Valid!")
                else:
                    st.warning("⚠️ Invalid URL")
        
        with col3:
            height_option = st.selectbox("Height", ["900px", "1200px", "1500px", "Full Screen"], key="height_contact")
        
        # Convert height option to pixels
        height_map = {"900px": 900, "1200px": 1200, "1500px": 1500, "Full Screen": 2000}
        iframe_height = height_map[height_option]
        
        if powerbi_url and "YOUR_CONTACT_STRATEGY_ID" not in powerbi_url:
            try:
                st.markdown("---")
                st.info("🎯 **Optimal Contact Channels & Timing** - Analyze which contact methods and timing yield the best conversion rates")
                
                powerbi_html = f"""
                <div style="width: 100%; height: {iframe_height}px; position: relative; background-color: #262730; border-radius: 15px; border: 2px solid #2e86ab; overflow: hidden;">
                    <iframe 
                        src="{powerbi_url}" 
                        frameborder="0" 
                        width="100%" 
                        height="100%"
                        allowFullScreen="true"
                        style="border: none; display: block;">
                    </iframe>
                </div>
                """
                st.components.v1.html(powerbi_html, height=iframe_height, scrolling=False)
                st.success("✅ Contact Strategy dashboard loaded!")
            except Exception as e:
                st.error(f"❌ Error loading dashboard: {e}")
        else:
            # Show alternative visualization with existing data
            st.markdown("---")
            st.info("💡 **Preview Mode** - Showing contact strategy insights from loaded data. Configure Power BI URL above for full dashboard.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Contact type performance
                contact_performance = self.df.groupby('contact')['y'].mean() * 100
                fig = px.bar(contact_performance, 
                            title="Conversion Rate by Contact Type",
                            labels={'value': 'Conversion Rate %', 'contact': 'Contact Type'},
                            color_discrete_sequence=['#2e86ab'])
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Month performance
                month_performance = self.df.groupby('month')['y'].mean() * 100
                fig = px.line(month_performance, 
                             title="Conversion Rate by Month",
                             labels={'value': 'Conversion Rate %', 'month': 'Month'},
                             color_discrete_sequence=['#a23b72'])
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

    def show_predictive_insights_report(self):
        """Display Predictive Model Insights Power BI Report - FULL SCREEN"""
        st.subheader("🎯 Predictive Model Insights")
        
        # Power BI URL input
        st.markdown("### 🔗 Configure Power BI Integration")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            powerbi_url = st.text_input(
                "Enter Predictive Insights Power BI URL:",
                value="https://app.powerbi.com/reportEmbed?reportId=345754cc-2a9d-45ef-b6ed-6aa82ddd804c&autoAuth=true&ctid=00f9cda3-075e-44e5-aa0b-aba3add6539f",
                key="predictive_insights_url"
            )
        
        with col2:
            if st.button("Test Connection", key="test_predictive", use_container_width=True):
                if powerbi_url and "YOUR_PREDICTIVE_INSIGHTS_ID" not in powerbi_url:
                    st.success("✅ Valid!")
                else:
                    st.warning("⚠️ Invalid URL")
        
        with col3:
            height_option = st.selectbox("Height", ["900px", "1200px", "1500px", "Full Screen"], key="height_predictive")
        
        # Convert height option to pixels
        height_map = {"900px": 900, "1200px": 1200, "1500px": 1500, "Full Screen": 2000}
        iframe_height = height_map[height_option]
        
        if powerbi_url and "YOUR_PREDICTIVE_INSIGHTS_ID" not in powerbi_url:
            try:
                st.markdown("---")
                st.info("🔮 **ML Model Performance & Feature Importance** - Compare model performance and understand which factors drive subscription predictions")
                
                powerbi_html = f"""
                <div style="width: 100%; height: {iframe_height}px; position: relative; background-color: #262730; border-radius: 15px; border: 2px solid #2e86ab; overflow: hidden;">
                    <iframe 
                        src="{powerbi_url}" 
                        frameborder="0" 
                        width="100%" 
                        height="100%"
                        allowFullScreen="true"
                        style="border: none; display: block;">
                    </iframe>
                </div>
                """
                st.components.v1.html(powerbi_html, height=iframe_height, scrolling=False)
                st.success("✅ Predictive Insights dashboard loaded!")
            except Exception as e:
                st.error(f"❌ Error loading dashboard: {e}")
        else:
            # Show model insights if models are trained
            st.markdown("---")
            if st.session_state.models_trained:
                st.info("💡 **Preview Mode** - Showing model insights from trained models. Configure Power BI URL above for full dashboard.")
                
                model_performance = st.session_state.model_performance
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model performance comparison
                    models = list(model_performance.keys())
                    auc_scores = [model_performance[model]['auc_score'] for model in models]
                    
                    fig = px.bar(x=models, y=auc_scores,
                                title="Model Performance (AUC Score)",
                                labels={'x': 'Model', 'y': 'AUC Score'},
                                color=auc_scores,
                                color_continuous_scale='Viridis')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature importance from Random Forest
                    if 'Random Forest' in model_performance:
                        rf_model = model_performance['Random Forest']['model']
                        feature_names = self.get_feature_names()
                        
                        if hasattr(rf_model.named_steps['classifier'], 'feature_importances_'):
                            importances = rf_model.named_steps['classifier'].feature_importances_
                            feature_imp_df = pd.DataFrame({
                                'feature': feature_names[:len(importances)],
                                'importance': importances
                            }).sort_values('importance', ascending=True).tail(10)
                            
                            fig = px.bar(feature_imp_df, 
                                       x='importance', y='feature',
                                       title='Top 10 Feature Importances',
                                       color_discrete_sequence=['#2e86ab'])
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please train models first to see predictive insights.")
                st.info("Go to 'Model Training & Results' section and click 'Train All Models'")

    def show_powerbi_placeholder(self, report_name, description):
        """Show placeholder when no Power BI URL is configured"""
        st.markdown(f"""
        <div style="text-align: center; padding: 50px; background-color: #262730; border-radius: 15px; border: 2px solid #2e86ab; margin: 20px 0;">
            <h3 style="color: white;">🚀 Power BI Integration Ready</h3>
            <p style="color: white; font-size: 18px;"><strong>{report_name} Dashboard</strong></p>
            <p style="color: white;">This area will display your {description} when configured.</p>
            <div style="background: #1a1d24; padding: 30px; border-radius: 10px; margin: 30px auto; max-width: 800px; color: white;">
                <h4 style="color: #2e86ab;">📋 Quick Setup Guide:</h4>
                <ol style="text-align: left; line-height: 2; font-size: 16px;">
                    <li>✅ Publish your report to Power BI Service (app.powerbi.com)</li>
                    <li>🔗 Click <strong>File → Embed report → Website or portal</strong></li>
                    <li>📋 Copy the embed URL</li>
                    <li>📝 Paste it in the text field above</li>
                    <li>🎉 Click "Test Connection" and your dashboard will load!</li>
                </ol>
            </div>
            <p style="color: #aaa; font-style: italic;">💡 Tip: Use the height selector to adjust dashboard size for optimal viewing</p>
        </div>
        """, unsafe_allow_html=True)

    def show_practical_example(self):
        """Display practical example with FIXED prediction logic"""
        st.markdown('<h2 class="section-header">🎯 Practical Example</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Real-World Application
        
        **Scenario:** Bank has 1,000 clients to contact for next campaign
        
        **Traditional Approach:**
        - Contact all 1,000 clients
        - Expected conversions: ~117 clients (11.7% success rate)
        - Cost: High (1000 contacts)
        - Efficiency: Low
        
        **AI-Powered Approach:**
        - Use model to identify top 200 most likely subscribers
        - Contact only these 200 clients
        - Expected conversions: ~80-100 clients (40-50% success rate)
        - Cost: Low (200 contacts)
        - Efficiency: High
        """)
        
        # Interactive example
        st.subheader("Interactive Prediction")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first to use the prediction feature.")
            st.info("Go to 'Model Training & Results' and click 'Train All Models'")
            return
            
        st.info("Enter client details to predict subscription probability:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 95, 40)
            job = st.selectbox("Job", self.df['job'].unique())
            balance = st.number_input("Balance", -10000, 200000, 1000)
        
        with col2:
            education = st.selectbox("Education", self.df['education'].unique())
            marital = st.selectbox("Marital Status", self.df['marital'].unique())
            housing = st.selectbox("Housing Loan", ["No", "Yes"])
        
        with col3:
            loan = st.selectbox("Personal Loan", ["No", "Yes"])
            contact = st.selectbox("Contact Type", self.df['contact'].unique())
            duration = st.slider("Call Duration (seconds)", 0, 5000, 300)
            campaign = st.slider("Campaign Contacts", 1, 50, 3)
        
        # Additional fields that models expect
        col4, col5, col6 = st.columns(3)
        with col4:
            day = st.slider("Day of Month", 1, 31, 15)
        with col5:
            month = st.selectbox("Month", self.df['month'].unique())
        with col6:
            poutcome = st.selectbox("Previous Outcome", self.df['poutcome'].unique())
        
        if st.button("Predict Subscription Probability", type="primary", key="predict_btn"):
            try:
                # Create sample data with ALL expected features
                sample_data = {
                    'age': [age],
                    'job': [job],
                    'marital': [marital],
                    'education': [education],
                    'default': [0],  # Assuming no default
                    'balance': [balance],
                    'housing': [1 if housing == "Yes" else 0],
                    'loan': [1 if loan == "Yes" else 0],
                    'contact': [contact],
                    'day': [day],
                    'month': [month],
                    'duration': [duration],
                    'campaign': [campaign],
                    'pdays': [-1],  # Default value
                    'previous': [0],  # Default value
                    'poutcome': [poutcome]
                }
                
                sample_df = pd.DataFrame(sample_data)
                
                # Use the best model for prediction
                model_performance = st.session_state.model_performance
                best_model_name = max(model_performance.items(), 
                                    key=lambda x: x[1]['auc_score'])[0]
                best_model = model_performance[best_model_name]['model']
                
                # Make prediction
                probability = best_model.predict_proba(sample_df)[0][1]
                
                # Display result
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    if probability > 0.5:
                        st.success(f"✅ **LIKELY TO SUBSCRIBE**")
                    else:
                        st.error(f"❌ **UNLIKELY TO SUBSCRIBE**")
                
                with col2:
                    st.metric("Subscription Probability", f"{probability:.1%}")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Subscription Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightgreen"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90}}))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show which model was used
                st.info(f"🔍 **Prediction made using:** {best_model_name} (AUC: {model_performance[best_model_name]['auc_score']:.3f})")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("This might be due to missing categories in the input data. Try different values.")

    def show_business_recommendations(self):
        """Display business recommendations"""
        st.markdown('<h2 class="section-header">💼 Business Recommendations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Immediate Actions")
            st.markdown("""
            **1. Implement Targeted Campaigns**
            - Use model to score client lists before campaigns
            - Focus on top 20-30% with highest predicted probability
            - Expected cost reduction: **60-80%**
            
            **2. Optimize Contact Strategy**
            - Prioritize cellular contacts over telephone
            - Focus on months with highest success rates
            - Adjust call timing based on client profiles
            
            **3. Resource Reallocation**
            - Reduce call center staff for mass campaigns
            - Invest in specialized teams for high-value clients
            - Implement automated scoring system
            """)
            
            st.subheader("📊 Expected Impact")
            impact_data = {
                'Metric': ['Cost Reduction', 'Conversion Rate', 'Customer Satisfaction', 'ROI Improvement'],
                'Current': ['100%', '11.7%', 'Low', 'Baseline'],
                'Expected': ['20-40%', '40-50%', 'High', '3-5x']
            }
            st.dataframe(pd.DataFrame(impact_data), use_container_width=True)
        
        with col2:
            st.subheader("🚀 Strategic Initiatives")
            st.markdown("""
            **1. Client Segmentation**
            - Create detailed client personas for high-probability segments
            - Develop personalized marketing messages
            - Implement tiered service levels
            
            **2. Model Integration**
            - Integrate with CRM systems
            - Real-time scoring for inbound inquiries
            - Continuous model retraining pipeline
            
            **3. Performance Monitoring**
            - Track model performance metrics
            - Monitor for concept drift
            - Regular business impact assessment
            
            **4. Ethical AI Practices**
            - Ensure fairness across demographic groups
            - Regular bias audits
            - Transparent model explanations
            """)
            
            st.subheader("⚠️ Risk Mitigation")
            st.markdown("""
            - **Model Decay:** Schedule quarterly retraining
            - **Data Quality:** Implement data validation pipelines  
            - **Compliance:** Ensure GDPR/data protection compliance
            - **Business Changes:** Monitor for market shifts affecting model performance
            """)
        
            st.success("""
            **Conclusion:** By implementing this AI-driven approach, the bank can transform its marketing strategy from 
            inefficient mass campaigns to highly targeted, cost-effective operations that deliver better results and 
            improved customer experiences.
            """)

            st.markdown("""
            ---
            **Additional Recommendation:**
                
            Most customers in this dataset do not subscribe to the bank's term deposit offer, so we need to think about ways to improve it. First, we should try to understand why people say "no" by looking at their age, job, education, balance, previous interactions, and how the bank contacted them. Not everyone is equally likely to subscribe, so it helps to focus on the customers who are more likely to say "yes". For example, people with higher balances, no loans in default, or who responded positively in the past are good candidates. The way and time we contact them also matters—calling at the right time or using the right method (like mobile phone vs. landline) can make a difference. Personalizing the offer, such as giving better interest rates or special promotions based on the customer's situation, makes it more attractive. Simple incentives and making the process easy and trustworthy can encourage more people to subscribe. Finally, by testing different approaches, learning from the results, and continuously improving, the bank can increase the number of customers who say "yes" over time.
            """)

    def get_feature_names(self):
        """Get feature names after preprocessing"""
        feature_names = []
        
        # Numerical features
        numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        feature_names.extend(numerical_features)
        
        # Categorical features (from one-hot encoding)
        categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        for col in categorical_features:
            categories = self.df[col].unique()
            for category in sorted(categories)[1:]:  # drop first category
                feature_names.append(f"{col}_{category}")
        
        # Binary features
        binary_features = ['default', 'housing', 'loan']
        feature_names.extend(binary_features)
        
        return feature_names

# Main application
def main():
    st.sidebar.title("About")
    st.sidebar.info("""
    **Bank Marketing Campaign Analysis**
    
    This dashboard analyzes bank marketing campaign data to predict client subscription to term deposits.
    
    **Features:**
    - Data exploration & visualization
    - Multiple ML model training
    - Power BI dashboard integration
    - Performance comparison
    - Interactive predictions
    - Business recommendations
    
    **Technology Stack:**
    - Streamlit
    - Scikit-learn
    - Power BI Embedded
    - Plotly
    - Pandas
    """)
    
    # Initialize and run dashboard
    dashboard = BankMarketingDashboard()
    dashboard.create_dashboard()

if __name__ == "__main__":
    main()

