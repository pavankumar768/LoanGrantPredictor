import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from utils.data_processing import preprocess_data, handle_missing_values
from utils.model_training import train_models, perform_hyperparameter_tuning
from utils.model_evaluation import evaluate_models, plot_model_comparison
from utils.prediction import make_prediction, explain_prediction
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Bank Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

def main():
    st.title("🏦 Bank Loan Eligibility Predictor")
    st.markdown("### Machine Learning-Based Loan Approval System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Data Upload & Training", "Model Comparison", "Loan Prediction", "About"]
    )
    
    if page == "Data Upload & Training":
        data_upload_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "Loan Prediction":
        prediction_page()
    else:
        about_page()

def data_upload_page():
    st.header("📊 Data Upload & Model Training")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your loan dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing loan application data"
    )
    
    # Option to use sample data
    col1, col2 = st.columns(2)
    with col1:
        use_sample = st.button("📋 Use Sample Dataset", help="Load a pre-built sample dataset")
    
    data = None
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
    
    elif use_sample:
        try:
            # Load sample data
            sample_path = "sample_data/loan_dataset.csv"
            if os.path.exists(sample_path):
                data = pd.read_csv(sample_path)
                st.success("✅ Sample dataset loaded successfully!")
            else:
                st.error("Sample dataset not found. Please upload your own data.")
                return
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return
    
    if data is not None:
        # Display data overview
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            if 'Loan_Status' in data.columns:
                approval_rate = (data['Loan_Status'] == 'Y').mean() * 100
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Display first few rows
        st.subheader("🔍 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data preprocessing section
        st.subheader("⚙️ Data Preprocessing")
        
        if st.button("🔄 Start Data Preprocessing", type="primary"):
            with st.spinner("Processing data..."):
                try:
                    # Handle missing values
                    data_clean = handle_missing_values(data)
                    
                    # Preprocess data
                    X_processed, y, preprocessor, feature_names = preprocess_data(data_clean)
                    
                    # Store in session state
                    st.session_state.X_processed = X_processed
                    st.session_state.y = y
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = feature_names
                    
                    st.success("✅ Data preprocessing completed!")
                    
                    # Show preprocessing results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Processed Features", X_processed.shape[1])
                    with col2:
                        st.metric("Training Samples", X_processed.shape[0])
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
                    return
        
        # Model training section
        if hasattr(st.session_state, 'X_processed'):
            st.subheader("🤖 Model Training")
            
            # Training options
            col1, col2 = st.columns(2)
            with col1:
                perform_tuning = st.checkbox("Perform Hyperparameter Tuning", value=True)
            with col2:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        # Train models
                        models, model_results = train_models(
                            st.session_state.X_processed, 
                            st.session_state.y,
                            perform_tuning=perform_tuning,
                            cv_folds=cv_folds
                        )
                        
                        # Store results
                        st.session_state.models = models
                        st.session_state.model_results = model_results
                        st.session_state.models_trained = True
                        
                        # Find best model
                        best_model_name = max(model_results.keys(), 
                                            key=lambda x: model_results[x]['cv_scores'].mean())
                        st.session_state.best_model = models[best_model_name]
                        st.session_state.best_model_name = best_model_name
                        
                        st.success("✅ Models trained successfully!")
                        st.info(f"🏆 Best performing model: **{best_model_name}**")
                        
                        # Display quick results
                        results_df = pd.DataFrame({
                            'Model': list(model_results.keys()),
                            'CV Score': [f"{results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}" 
                                       for results in model_results.values()],
                            'Training Time': [f"{results['training_time']:.2f}s" 
                                            for results in model_results.values()]
                        })
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")

def model_comparison_page():
    st.header("📊 Model Comparison & Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Data Upload & Training' page.")
        return
    
    try:
        # Model performance comparison
        st.subheader("🏆 Model Performance Comparison")
        
        # Create performance comparison chart
        model_names = list(st.session_state.model_results.keys())
        cv_means = [st.session_state.model_results[name]['cv_scores'].mean() for name in model_names]
        cv_stds = [st.session_state.model_results[name]['cv_scores'].std() for name in model_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=cv_means,
            error_y=dict(type='data', array=cv_stds),
            text=[f"{mean:.4f}" for mean in cv_means],
            textposition='auto',
            marker_color=['gold' if name == st.session_state.best_model_name else 'lightblue' 
                         for name in model_names]
        ))
        
        fig.update_layout(
            title="Cross-Validation Scores Comparison",
            xaxis_title="Models",
            yaxis_title="Accuracy Score",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive model evaluation metrics
        st.subheader("📈 Comprehensive Model Evaluation")
        
        # Evaluate models on training data for comprehensive metrics
        evaluation_results = evaluate_models(st.session_state.models, 
                                           st.session_state.X_processed, 
                                           st.session_state.y)
        
        # Create metrics dataframe
        metrics_data = []
        for model_name in model_names:
            metrics = evaluation_results[model_name]['metrics']
            cv_results = st.session_state.model_results[model_name]
            metrics_data.append({
                'Model': model_name,
                'CV Accuracy': f"{cv_results['cv_scores'].mean():.4f} ± {cv_results['cv_scores'].std():.4f}",
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}",
                'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else 'N/A',
                'Training Time': f"{cv_results['training_time']:.2f}s"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Create metrics comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision vs Recall scatter plot
            precisions = [evaluation_results[name]['metrics'].get('precision', 0) for name in model_names]
            recalls = [evaluation_results[name]['metrics'].get('recall', 0) for name in model_names]
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=precisions,
                y=recalls,
                mode='markers+text',
                text=model_names,
                textposition="top center",
                marker=dict(size=10, color=['gold' if name == st.session_state.best_model_name else 'lightblue' 
                                          for name in model_names])
            ))
            fig_scatter.update_layout(
                title="Precision vs Recall",
                xaxis_title="Precision",
                yaxis_title="Recall",
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # F1-Score comparison
            f1_scores = [evaluation_results[name]['metrics'].get('f1_score', 0) for name in model_names]
            
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                x=model_names,
                y=f1_scores,
                text=[f"{score:.4f}" for score in f1_scores],
                textposition='auto',
                marker_color=['gold' if name == st.session_state.best_model_name else 'lightblue' 
                             for name in model_names]
            ))
            fig_f1.update_layout(
                title="F1-Score Comparison",
                xaxis_title="Models",
                yaxis_title="F1-Score",
                showlegend=False
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Feature importance analysis
        st.subheader("🎯 Feature Importance Analysis")
        
        # Get feature importance from the best model
        if hasattr(st.session_state.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': st.session_state.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"Top 15 Feature Importances - {st.session_state.best_model_name}"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance table
            st.dataframe(importance_df, use_container_width=True)
        
        else:
            st.info("Feature importance not available for the selected model type.")
    
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")

def prediction_page():
    st.header("🔮 Loan Eligibility Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Data Upload & Training' page.")
        return
    
    st.subheader("📝 Loan Application Form")
    
    # Create input form
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Personal Information**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            
        with col2:
            st.markdown("**Financial Information**")
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=100)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150000, step=1000)
            loan_term = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
            credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Poor")
        
        col3, col4 = st.columns(2)
        with col3:
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        with col4:
            loan_type = st.selectbox("Loan Type", ["Home", "Education"])
            
        # Additional fields based on loan type
        if loan_type == "Home":
            property_value = st.number_input("Property Value ($)", min_value=0, value=200000, step=5000)
            course_type = None
            academic_performance = None
        else:
            property_value = None
            course_type = st.selectbox("Course Type", ["Engineering", "Medical", "MBA", "Law", "Other"])
            academic_performance = st.slider("Academic Performance (GPA)", 2.0, 4.0, 3.5, 0.1)
        
        submitted = st.form_submit_button("🔍 Predict Loan Eligibility", type="primary")
    
    if submitted:
        try:
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area,
                'Loan_Type': loan_type
            }
            
            # Add conditional fields
            if loan_type == "Home":
                input_data['Property_Value'] = property_value
            else:
                input_data['Course_Type'] = course_type
                input_data['Academic_Performance'] = academic_performance
            
            # Make prediction
            prediction, probability, explanation = make_prediction(
                input_data,
                st.session_state.best_model,
                st.session_state.preprocessor,
                st.session_state.feature_names
            )
            
            # Display results
            if prediction is not None and probability is not None:
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.success("✅ **APPROVED**")
                        st.success("Congratulations! Your loan application is likely to be approved.")
                    else:
                        st.error("❌ **REJECTED**")
                        st.error("Unfortunately, your loan application is likely to be rejected.")
                
                with col2:
                    # Find the positive class index (approval = 1)
                    pos_class_idx = np.where(st.session_state.best_model.classes_ == 1)[0][0]
                    approval_prob = probability[pos_class_idx]
                    st.metric("Approval Probability", f"{approval_prob:.2%}")
                
                with col3:
                    st.metric("Model Used", st.session_state.best_model_name)
                
                # Explanation
                st.subheader("💡 Decision Explanation")
                
                if explanation is not None:
                    # Create explanation visualization
                    feature_impact = pd.DataFrame({
                        'Feature': explanation['feature_names'],
                        'Impact': explanation['shap_values'],
                        'Value': explanation['feature_values']
                    })
                    
                    # Sort by absolute impact
                    feature_impact['abs_impact'] = abs(feature_impact['Impact'])
                    feature_impact = feature_impact.sort_values('abs_impact', ascending=False).head(10)
                    
                    # Create horizontal bar chart
                    colors = ['green' if x > 0 else 'red' for x in feature_impact['Impact']]
                    
                    fig = go.Figure(go.Bar(
                        x=feature_impact['Impact'],
                        y=feature_impact['Feature'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{val}" for val in feature_impact['Value']],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Top 10 Factors Influencing the Decision",
                        xaxis_title="Impact on Prediction (SHAP Values)",
                        yaxis_title="Features",
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation text
                    st.markdown("**How to interpret this chart:**")
                    st.markdown("- **Green bars** (positive values): Factors that increase approval probability")
                    st.markdown("- **Red bars** (negative values): Factors that decrease approval probability")
                    st.markdown("- **Longer bars**: More influential factors")
                else:
                    st.info("Detailed explanation not available for this model type.")
                
                # Recommendations
                st.subheader("💼 Recommendations")
                
                if prediction == 0:  # Rejected
                    st.markdown("**To improve your chances of approval:**")
                    # Find the positive class index (approval = 1) for probability threshold
                    pos_class_idx = np.where(st.session_state.best_model.classes_ == 1)[0][0]
                    approval_prob = probability[pos_class_idx]
                    if approval_prob < 0.3:
                        st.markdown("- Consider improving your credit history")
                        st.markdown("- Increase your income or add a co-applicant")
                        st.markdown("- Reduce the loan amount requested")
                        st.markdown("- Consider a longer loan term to reduce monthly payments")
                    else:
                        st.markdown("- Your application is borderline. Consider minor improvements to financial profile")
                        st.markdown("- Provide additional documentation to support your application")
                else:  # Approved
                    st.markdown("**Your application looks strong! Consider:**")
                    st.markdown("- Reviewing loan terms and interest rates")
                    st.markdown("- Preparing necessary documentation")
                    st.markdown("- Shopping around for the best rates from different lenders")
            else:
                st.error("Unable to make prediction. Please check your input data and try again.")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def about_page():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🏦 Bank Loan Eligibility Predictor
    
    This machine learning web application helps predict bank loan eligibility for both home loans and education loans.
    
    ### 🎯 Features
    
    - **Data Upload & Preprocessing**: Upload your own CSV dataset or use the sample dataset
    - **Multi-Model Training**: Compare Logistic Regression, Random Forest, and XGBoost models
    - **Model Evaluation**: Comprehensive evaluation with accuracy, precision, recall, F1-score, and ROC-AUC
    - **Hyperparameter Tuning**: Automated optimization for best performance
    - **Real-time Predictions**: Interactive form for instant loan eligibility predictions
    - **SHAP Explanations**: Understand which factors influence loan decisions
    - **Feature Importance**: Visualize the most important factors in loan approval
    
    ### 📊 Supported Loan Types
    
    1. **Home Loans**: Includes property value considerations
    2. **Education Loans**: Includes academic performance and course type factors
    
    ### 🤖 Machine Learning Models
    
    - **Logistic Regression**: Linear model for baseline predictions
    - **Random Forest**: Ensemble method with feature importance
    - **XGBoost**: Gradient boosting for high performance
    
    ### 📈 Model Evaluation Metrics
    
    - **Accuracy**: Overall prediction correctness
    - **Precision**: Positive prediction accuracy
    - **Recall**: True positive detection rate
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under the receiver operating characteristic curve
    
    ### 🔧 Technical Stack
    
    - **Frontend**: Streamlit
    - **ML Libraries**: Scikit-learn, XGBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Explainability**: SHAP
    
    ### 📝 Usage Instructions
    
    1. **Upload Data**: Go to 'Data Upload & Training' and upload your dataset or use sample data
    2. **Train Models**: Preprocess data and train the ML models
    3. **Compare Models**: View model performance in 'Model Comparison'
    4. **Make Predictions**: Use 'Loan Prediction' to test individual applications
    
    ### ⚠️ Important Notes
    
    - This application is for demonstration purposes
    - Real loan decisions involve many additional factors
    - Always consult with financial advisors for actual loan applications
    - Model predictions should be used as guidance only
    
    ### 📞 Support
    
    For technical issues or questions about the application, please refer to the documentation or contact the development team.
    """)

if __name__ == "__main__":
    main()
