import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="OptiFlow-RL | AI Traffic Control", layout="wide")

# --- NAVIGATION ---
tabs = st.tabs(["Home", "Intersection Layout", "MDP Architecture", "Evaluation"])

# --- TAB 1: HOME ---
with tabs[0]:
    st.title("🚦 OptiFlow-RL")
    st.subheader("B.Tech Artificial Intelligence Project")
    st.markdown("""
    **Objective:** To reduce urban traffic congestion using Deep Reinforcement Learning.
    By treating an intersection as an MDP, our agent learns to clear queues faster than 
    traditional fixed-time systems.
    """)

# --- TAB 2: ENVIRONMENT ---
with tabs[1]:
    st.header("📍 Simulation Environment")
    st.write("Layout: 4-Way Single Lane Intersection (Junction A0)")
    # If you have an image of your netedit layout, display it here:
    # st.image("env/layout_screenshot.png")
    st.code(open("env/network.net.xml", "r").read()[:500] + "...", language="xml")

# --- TAB 3: MDP ARCHITECT (YOUR SECTION) ---
with tabs[2]:
    st.header("🧠 MDP Brain (Architect's View)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Reward Shaping")
        w_wait = st.slider("Waiting Time Penalty", 0.0, 1.0, 0.8)
        w_queue = st.slider("Queue Length Penalty", 0.0, 1.0, 0.2)
        st.latex(r"R = -(w_{wait} \cdot T_{wait} + w_{queue} \cdot L_{queue})")
    
    with col2:
        st.write("### State Normalization")
        raw_val = st.number_input("Raw Vehicle Count", 0, 25, 10)
        norm_val = raw_val / 25
        st.metric("Normalized State Input", f"{norm_val:.2f}")

# --- TAB 4: RESULTS ---
with tabs[3]:
    st.header("📊 Final Performance")
    if st.button("Load Latest Benchmark"):
        # This will load the Analyst's CSV data
        try:
            results = pd.read_csv("logs/benchmark_results.csv")
            st.line_chart(results)
        except:
            st.error("Benchmark data not found. Run evaluate.py first.")
