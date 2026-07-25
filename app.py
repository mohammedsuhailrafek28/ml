import streamlit as st
from src.utils.config import DISEASES, DISCLAIMER
from src.prediction.predictor import predict
st.set_page_config(page_title="Medical AI Suite", page_icon="🩺", layout="wide")
st.title("🩺 Medical AI Suite")
st.warning(DISCLAIMER)
page=st.sidebar.radio("Navigate",["Home","Disease prediction","Model comparison","Dataset insights"])
if page=="Home":
    st.header("University machine-learning project"); st.write("A unified educational workflow for liver, heart, diabetes, chronic kidney, and Parkinson's risk-pattern datasets.")
    st.subheader("Workflow"); st.write("Public data → documented cleaning → leakage-safe preprocessing → cross-validated models → cautious explanations.")
elif page=="Disease prediction":
    disease=st.selectbox("Disease module",list(DISEASES),format_func=lambda x:DISEASES[x].title); c=DISEASES[disease]
    st.caption("Enter all fields. Values are dataset fields, not a clinical intake form.")
    cols=st.columns(3); values={}
    for i,f in enumerate(c.features): values[f]=cols[i%3].number_input(f.replace('_',' ').title(),value=0.0,key=f)
    if st.button("Estimate pattern"):
        try:
            result=predict(disease,values); st.success(f"Predicted class: {result['prediction']}")
            if result['risk_probability'] is not None: st.metric("Estimated positive probability",f"{result['risk_probability']:.0%}")
            st.info(result['interpretation']); st.warning(DISCLAIMER)
        except FileNotFoundError: st.error("No trained model found. Add the documented dataset, then run python -m src.training.train_all.")
        except Exception as e: st.error(str(e))
elif page=="Model comparison":
    import pandas as pd
    for k,c in DISEASES.items():
        p=c.model_dir.parent.parent/"reports"/"model_results"/f"{k}_metrics.csv"
        if p.exists(): st.subheader(c.title); st.dataframe(pd.read_csv(p),use_container_width=True)
        else: st.info(f"{c.title}: train this module to display measured metrics.")
else:
    st.header("Dataset insights"); st.write("Charts are generated from the actual public datasets after cleaning and are stored in reports/figures.")
