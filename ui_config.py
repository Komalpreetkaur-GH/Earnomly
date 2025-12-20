import streamlit as st

# --- Custom SVG Icons ---
SVG_ICONS = {
    "logo": """
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="32" height="32" rx="8" fill="url(#logo_gradient)"/>
            <path d="M8 16C8 11.5817 11.5817 8 16 8V8C20.4183 8 24 11.5817 24 16V24H16C11.5817 24 8 20.4183 8 16V16Z" fill="white" fill-opacity="0.9"/>
            <circle cx="16" cy="16" r="3" fill="url(#logo_gradient)"/>
            <defs>
                <linearGradient id="logo_gradient" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#6366F1"/>
                    <stop offset="1" stop-color="#8B5CF6"/>
                </linearGradient>
            </defs>
        </svg>
    """,
    "overview": """
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="2" width="7" height="7" rx="2" stroke="currentColor" stroke-width="1.5"/>
            <rect x="11" y="2" width="7" height="7" rx="2" stroke="currentColor" stroke-width="1.5"/>
            <rect x="2" y="11" width="7" height="7" rx="2" stroke="currentColor" stroke-width="1.5"/>
            <rect x="11" y="11" width="7" height="7" rx="2" stroke="currentColor" stroke-width="1.5"/>
        </svg>
    """,
    "analytics": """
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 17V10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M8 17V6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M13 17V8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M18 17V3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    """,
    "intelligence": """
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"/>
            <path d="M10 6V10L13 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    """,
    "clusters": """
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="6" cy="6" r="3" stroke="currentColor" stroke-width="1.5"/>
            <circle cx="14" cy="6" r="3" stroke="currentColor" stroke-width="1.5"/>
            <circle cx="10" cy="14" r="3" stroke="currentColor" stroke-width="1.5"/>
            <path d="M8 8L9 12" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>
            <path d="M12 8L11 12" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>
        </svg>
    """,
    "predictor": """
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M10 10L14 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="10" cy="10" r="2" fill="currentColor"/>
            <path d="M5 17H15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    """,
    "population": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="9" cy="7" r="3" stroke="currentColor" stroke-width="1.5"/>
            <circle cx="17" cy="7" r="2" stroke="currentColor" stroke-width="1.5"/>
            <path d="M3 21V18C3 15.7909 4.79086 14 7 14H11C13.2091 14 15 15.7909 15 18V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M17 14C19.2091 14 21 15.7909 21 18V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    """,
    "features": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.5"/>
            <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.5"/>
            <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.5"/>
            <path d="M17.5 14V21M14 17.5H21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
    """,
    "clean": """
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/>
            <path d="M8 12L11 15L16 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    """,
    "arrow_up": """
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 12V4M8 4L4 8M8 4L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    """,
    "status_online": """
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="4" cy="4" r="4" fill="#10B981"/>
        </svg>
    """
}

def get_svg(name, color=None):
    """Get SVG icon by name with optional color override."""
    svg = SVG_ICONS.get(name, "")
    if color:
        svg = svg.replace('currentColor', color)
    return svg

def apply_styles():
    st.markdown("""
        <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

        /* Root Variables - Minimalist Fintech Theme */
        :root {
            --bg-primary: #0A0A0B;
            --bg-secondary: #111113;
            --bg-card: rgba(17, 17, 19, 0.8);
            --bg-card-hover: rgba(24, 24, 27, 0.9);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-hover: rgba(255, 255, 255, 0.12);
            --text-primary: #FAFAFA;
            --text-secondary: #71717A;
            --text-muted: #52525B;
            --accent-primary: #6366F1;
            --accent-secondary: #8B5CF6;
            --accent-success: #10B981;
            --accent-info: #3B82F6;
            --font-display: 'Outfit', sans-serif;
            --font-body: 'Inter', sans-serif;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --transition-fast: 0.15s ease;
            --transition-smooth: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Base Background */
        .stApp {
            background: var(--bg-primary);
            background-image: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%);
        }

        /* Global Typography */
        html, body, [class*="css"] {
            font-family: var(--font-body);
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Block Container */
        .block-container {
            padding: 2rem 3rem 6rem 3rem !important;
            max-width: 1400px;
        }

        /* Headers - Clean Typography */
        h1 {
            font-family: var(--font-display) !important;
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: var(--text-primary) !important;
            margin-bottom: 0.5rem !important;
        }

        h2 {
            font-family: var(--font-display) !important;
            font-size: 1.125rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            color: var(--text-primary) !important;
            margin-bottom: 1rem !important;
        }

        h3 {
            font-family: var(--font-body) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Sidebar - Ultra Clean */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1rem !important;
        }

        /* Hide default Streamlit nav */
        div[data-testid="stSidebarNav"] {
            display: none;
        }

        /* Sidebar Brand */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0 0.75rem;
            margin-bottom: 2rem;
        }

        .brand-text {
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        /* Navigation Category Label */
        .nav-category {
            font-size: 0.6875rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em !important;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            padding: 0 0.75rem;
            margin: 1.5rem 0 0.75rem 0;
        }

        /* Radio Navigation - Clean Minimal Style */
        div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 0 0.5rem;
        }

        div[role="radiogroup"] > label {
            background: transparent !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 10px 12px !important;
            color: var(--text-secondary) !important;
            transition: var(--transition-smooth) !important;
            cursor: pointer !important;
            font-size: 0.875rem !important;
            font-weight: 450 !important;
        }

        div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.04) !important;
            color: var(--text-primary) !important;
        }

        div[role="radiogroup"] > label[data-selected="true"] {
            background: rgba(99, 102, 241, 0.1) !important;
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        div[role="radiogroup"] input {
            display: none;
        }

        div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
            font-size: 0.875rem !important;
            margin: 0 !important;
        }

        /* Glass Cards - Subtle */
        .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.25rem;
            transition: var(--transition-smooth);
        }

        .glass-card:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover {
            background: var(--bg-card-hover) !important;
            border-color: var(--border-hover) !important;
        }

        /* Metric Cards - Minimalist Fintech */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            transition: var(--transition-smooth);
        }

        .metric-card:hover {
            border-color: var(--border-hover);
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-family: var(--font-display);
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        .metric-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.6875rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-success);
        }

        .metric-badge.info {
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent-info);
        }

        .metric-badge.accent {
            background: rgba(139, 92, 246, 0.1);
            color: var(--accent-secondary);
        }

        .metric-icon {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: var(--radius-sm);
            background: rgba(99, 102, 241, 0.1);
            color: var(--accent-primary);
            margin-bottom: 0.75rem;
        }

        /* Progress Bar - Thin Line */
        .progress-bar {
            width: 100%;
            height: 3px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 0.75rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.6s ease;
        }

        .progress-fill.success { background: var(--accent-success); }
        .progress-fill.info { background: var(--accent-info); }
        .progress-fill.accent { background: var(--accent-secondary); }

        /* Buttons - Clean Fintech Style */
        .stButton > button {
            background: var(--accent-primary) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.625rem 1.25rem !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            transition: var(--transition-smooth) !important;
        }

        .stButton > button:hover {
            background: #5558E3 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Inputs - Minimal */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            font-size: 0.875rem !important;
            transition: var(--transition-fast);
        }

        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus-within {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        }

        /* Tabs - Fintech Pill Style */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255, 255, 255, 0.02);
            padding: 4px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-subtle);
        }

        .stTabs [data-baseweb="tab"] {
            height: 34px;
            border-radius: var(--radius-sm);
            background: transparent;
            color: var(--text-secondary);
            font-weight: 450;
            font-size: 0.8125rem;
            border: none !important;
            transition: var(--transition-fast);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.08) !important;
            color: var(--text-primary) !important;
            font-weight: 500;
        }

        /* Slider - Minimal */
        .stSlider > div > div > div > div {
            background: var(--accent-primary) !important;
        }

        .stSlider > div > div > div > div > div {
            background: var(--accent-primary) !important;
            border: 2px solid white !important;
        }

        /* DataFrames - Clean */
        .stDataFrame {
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        .stDataFrame [data-testid="stDataFrameContainer"] {
            background: var(--bg-card);
        }

        /* Plotly Charts */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }

        /* Status Footer */
        .status-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 14px;
        }

        .status-label {
            font-size: 0.625rem;
            font-weight: 500;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .status-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            background: var(--accent-success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-success);
        }

        .status-text {
            font-size: 0.8125rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .version-tag {
            font-size: 0.6875rem;
            color: var(--text-muted);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.12);
        }

        /* Alerts - Clean */
        .stAlert {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border-subtle) !important;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: var(--accent-primary) !important;
        }

        /* Section Divider */
        .section-divider {
            height: 1px;
            background: var(--border-subtle);
            margin: 2rem 0;
        }

        /* Intro Text */
        .intro-text {
            font-size: 0.9375rem;
            color: var(--text-secondary);
            line-height: 1.6;
            max-width: 600px;
        }

        /* Footer */
        .footer-text {
            font-size: 0.6875rem;
            color: var(--text-muted);
            text-align: center;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar_brand():
    """Render the sidebar brand with custom SVG logo."""
    logo_svg = '<svg width="32" height="32" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="5" width="90" height="90" rx="20" stroke="#6366F1" stroke-width="4"/><circle cx="50" cy="32" r="10" stroke="#6366F1" stroke-width="4"/><rect x="26" y="55" width="12" height="20" rx="2" fill="#6366F1"/><rect x="44" y="48" width="12" height="27" rx="2" fill="#8B5CF6"/><rect x="62" y="41" width="12" height="34" rx="2" fill="#6366F1"/></svg>'
    st.markdown(f'<div class="sidebar-brand">{logo_svg}<span class="brand-text">Earnomly</span></div>', unsafe_allow_html=True)

def render_metric_card(label, value, badge_text=None, badge_type="success", progress=None, icon_name=None):
    """Render a minimalist metric card with optional icon, badge, and progress bar."""
    # Inline SVG icons
    icons = {
        "population": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="9" cy="7" r="3" stroke="#6366F1" stroke-width="1.5"/><circle cx="17" cy="7" r="2" stroke="#6366F1" stroke-width="1.5"/><path d="M3 21V18C3 15.79 4.79 14 7 14H11C13.21 14 15 15.79 15 18V21" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round"/><path d="M17 14C19.21 14 21 15.79 21 18V21" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round"/></svg>',
        "features": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="7" height="7" rx="1" stroke="#6366F1" stroke-width="1.5"/><rect x="14" y="3" width="7" height="7" rx="1" stroke="#6366F1" stroke-width="1.5"/><rect x="3" y="14" width="7" height="7" rx="1" stroke="#6366F1" stroke-width="1.5"/><path d="M17.5 14V21M14 17.5H21" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round"/></svg>',
        "clean": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="#6366F1" stroke-width="1.5"/><path d="M8 12L11 15L16 9" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    }
    
    icon_html = f'<div class="metric-icon">{icons.get(icon_name, "")}</div>' if icon_name and icon_name in icons else ''
    badge_html = f'<span class="metric-badge {badge_type}">{badge_text}</span>' if badge_text else ''
    progress_html = f'<div class="progress-bar"><div class="progress-fill {badge_type}" style="width:{progress}%"></div></div>' if progress is not None else ''
    
    return f'<div class="metric-card">{icon_html}<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px"><span class="metric-label">{label}</span>{badge_html}</div><div class="metric-value">{value}</div>{progress_html}</div>'

def render_status_footer():
    """Render the sidebar status footer."""
    return '<div class="status-card"><div class="status-label">System Status</div><div class="status-content"><div class="status-indicator"><div class="status-dot"></div><span class="status-text">Online</span></div><span class="version-tag">v1.2.0</span></div></div><div class="footer-text">2024 Earnomly</div>'
