import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        /* Import Google Fonts - Using Inter for a clean, engineered look */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* General App Styling */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #F1F5F9; /* Slate 100 */
            background-color: #0F172A; /* Slate 900 */
        }

        /* Solid Professional Background - No Gradients */
        .stApp {
            background-color: #0F172A;
        }

        /* Sidebar Styling - Matte Dark */
        [data-testid="stSidebar"] {
            background-color: #1E293B; /* Slate 800 */
            border-right: 1px solid #334155; /* Slate 700 */
        }
        
        [data-testid="stSidebar"] * {
            color: #94A3B8 !important; /* Slate 400 */
        }
        
        [data-testid="stSidebarNav"] div[data-testid="stRadio"] label {
            font-weight: 500;
        }

        /* Content Cards - Clean, Solid Surface */
        .glass-card {
            background-color: #1E293B; /* Slate 800 */
            border: 1px solid #334155; /* Slate 700 */
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Headers - White and Clean */
        h1, h2, h3 {
            color: #F8FAFC !important; /* Slate 50 */
            font-weight: 600 !important;
            letter-spacing: -0.025em;
        }
        
        h1 { font-size: 2.25rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }

        /* Buttons - Professional Blue, No Gradients */
        .stButton>button {
            width: 100%;
            background-color: #3B82F6; /* Blue 500 */
            color: white;
            border: 1px solid #2563EB;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #2563EB; /* Blue 600 */
            border-color: #1D4ED8;
            transform: none; /* No jumpy animation */
            box-shadow: none;
        }

        /* Inputs - Clean Borders */
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
            background-color: #1E293B;
            color: #F1F5F9;
            border: 1px solid #475569; /* Slate 600 */
            border-radius: 6px;
        }
        
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 1px #3B82F6;
        }
        
        /* Metric Styling - Clean Numbers */
        [data-testid="stMetricValue"] {
            font-size: 1.875rem !important; /* 30px */
            font-weight: 600 !important;
            color: #F8FAFC !important;
            background: none !important;
            -webkit-text-fill-color: initial !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem !important;
            color: #94A3B8 !important; /* Slate 400 */
        }
        
        /* Plotly Chart Backgrounds */
        .js-plotly-plot .plotly .main-svg {
            background: rgba(0,0,0,0) !important;
        }

        </style>
    """, unsafe_allow_html=True)
