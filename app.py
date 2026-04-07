"""
ML Explorer — navigation router
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

pg = st.navigation([
    st.Page("home.py",                              title="Home",                    icon="🏠"),
    st.Page("pages/1_Supervised_Learning.py",       title="Supervised Learning",     icon="📊"),
    st.Page("pages/2_Unsupervised_Learning.py",     title="Unsupervised Learning",   icon="🔍"),
    st.Page("pages/3_Probability_Regression.py",    title="Probability & Regression",icon="📈"),
    st.Page("pages/4_Model_Playground.py",          title="Model Playground",        icon="🎮"),
])
pg.run()
