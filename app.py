import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ui_config

# Set page configuration
st.set_page_config(page_title="Adult Income Analysis", layout="wide", page_icon=None)

# Custom CSS for better styling
ui_config.apply_styles()

# Title and Introduction
# Title and Introduction
st.title("Adult Income Prediction & Analysis")
st.markdown("""
<div class="glass-card">
    <p><strong>Welcome</strong> · This professional dashboard explores the <strong>Adult Income Dataset</strong> to analyze socioeconomic factors and predict income levels (>50K vs <=50K).</p>
</div>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
               'hours-per-week', 'native-country', 'income']
    try:
        df = pd.read_csv('adultData/adult.data', names=columns, na_values='?', skipinitialspace=True)
        # Impute missing values
        for col in ['workclass', 'occupation', 'native-country']:
            df[col] = df[col].fillna(df[col].mode()[0])
        df.drop_duplicates(inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'adultData/adult.data' not found. Please ensure the dataset is in the correct directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Preprocessing Helper ---
def prepare_data_for_model(df):
    le = LabelEncoder()
    df_model = df.copy()
    # Encode target
    df_model['income_encoded'] = le.fit_transform(df_model['income'])
    df_model = df_model.drop('income', axis=1)
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    
    X = df_encoded.drop('income_encoded', axis=1)
    y = df_encoded['income_encoded']
    
    return X, y, le, X.columns

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", 
    ["Data Overview", "Visualizations", "Model & Prediction", "Unsupervised Learning", "Predict Your Income"])

# --- 1. Data Overview ---
if options == "Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Features", df.shape[1]-1)

    st.subheader("Sample System Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0], use_container_width=True)

# --- 2. Visualizations (Plotly) ---
elif options == "Visualizations":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Distribution")
        counts = df['income'].value_counts()
        fig = px.pie(names=counts.index, values=counts.values, hole=0.4, 
                     title="Income Class Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Age vs Income")
        fig = px.box(df, x='income', y='age', color='income', 
                     title="Age Distribution by Income", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Interactive Correlation Heatmap")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                         title="Correlation Matrix (Numerical Features)")
    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Workclass vs Income")
    fig_bar = px.histogram(df, x='workclass', color='income', barmode='group', 
                           title="Income Distribution by Workclass", color_discrete_sequence=px.colors.qualitative.Safe)
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
    st.plotly_chart(fig_bar, use_container_width=True)

# --- 3. Model & Prediction ---
elif options == "Model & Prediction":
    st.header("Machine Learning Models")
    
    X, y, le, model_columns = prepare_data_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Training Data Shape: {X_train.shape}")
    st.write(f"Testing Data Shape: {X_test.shape}")
    
    col_settings, col_train = st.columns([1, 2])
    
    with col_settings:
        st.subheader("Settings")
        model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
        
        params = {}
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees (n_estimators)", 10, 200, 50, 10)
            max_depth = st.slider("Max Depth", 5, 50, 20)
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
        else:
            C_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            params['C'] = C_val

        train_btn = st.button("Start Training")

    if train_btn:
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                               max_depth=params['max_depth'], random_state=42)
            else:
                model = LogisticRegression(C=params['C'], max_iter=1000)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"Model Trained successfully. Accuracy: {acc:.4f}")
            
            # Metrics
            st.subheader("Evaluation Metrics")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['<=50K', '>50K'], y=['<=50K', '>50K'],
                               title="Confusion Matrix")
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Save model components for Prediction tab (using session state would be better but this is simple)
            st.session_state['model'] = model
            st.session_state['model_columns'] = model_columns
            st.session_state['le'] = le

# --- 4. Unsupervised Learning ---
elif options == "Unsupervised Learning":
    st.header("Unsupervised Learning")
    
    # Preprocessing
    X, y, le, _ = prepare_data_for_model(df)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("1. K-Means Clustering")
    
    # Elbow Method
    if st.checkbox("Show Elbow Method Plot"):
        wcss = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        fig_elbow = px.line(x=k_range, y=wcss, markers=True, 
                            labels={'x':'Number of Clusters', 'y':'WCSS'}, title="Elbow Method")
        fig_elbow.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    k_val = st.slider("Select K (Clusters)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    st.subheader("2. PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters.astype(str)
    
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                         title=f"PCA Visualization (K={k_val})", opacity=0.7)
    fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.info(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# --- 5. Predict Your Income (New Feature) ---
elif options == "Predict Your Income":
    st.header("Predict Income Level")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model in the 'Model & Prediction' tab first.")
    else:
        st.write("Enter your details below to see what the model predicts.")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 17, 90, 30)
                workclass = st.selectbox("Workclass", df['workclass'].unique())
                education = st.selectbox("Education", df['education'].unique())
                marital_status = st.selectbox("Marital Status", df['marital-status'].unique())
                occupation = st.selectbox("Occupation", df['occupation'].unique())
                relationship = st.selectbox("Relationship", df['relationship'].unique())
                race = st.selectbox("Race", df['race'].unique())
            
            with col2:
                sex = st.selectbox("Sex", df['sex'].unique())
                capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
                capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
                hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
                native_country = st.selectbox("Native Country", df['native-country'].unique())
                
                # Hidden/Default fields (assumed or calculated)
                education_num = 10 # Simplifying for UI, ideally mapped from education
                fnlwgt = df['fnlwgt'].mean() # Use system average
            
            submit = st.form_submit_button("Predict")
            
        if submit:
             # Create a DataFrame for user input
            input_data = {
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [fnlwgt],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # One-Hot Encode and Align Columns
            input_encoded = pd.get_dummies(input_df)
            
            # Reindex to match training columns, filling missing with 0
            model_columns = st.session_state['model_columns']
            input_ready = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Predict
            model = st.session_state['model']
            prediction = model.predict(input_ready)
            le = st.session_state['le']
            result = le.inverse_transform(prediction)[0]
            
            if result == ">50K":
                st.success(f"Prediction: {result}")
            else:
                st.info(f"Prediction: {result}")
