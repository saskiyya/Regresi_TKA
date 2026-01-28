import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Prediksi Nilai TKA"
)

model = joblib.load("model.joblib")

st.title("Prediksi Nilai TKA")
st.markdown("Aplikasi machine learning regression untuk memprediksi nilai TKA")

jam_belajar_per_hari = st.slider("Jam Belajar Per Hari", 1.0, 10.0, 5.0)
persen_kehadiran = st.slider("Persen Kehadiran", 80.0, 100.0, 90.0)
bimbel = st.pills("Bimbel", ["ya","tidak"], default="ya")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[jam_belajar_per_hari, persen_kehadiran, bimbel]], 
							columns=["jam_belajar_per_hari", "persen_kehadiran", "bimbel"])
	
	prediksi = model.predict(data_baru)[0]
	prediksi = prediksi.clip(0, 100)
	
	st.success(f"Model memprediksi nilai TKA : **{prediksi:.0f}**")
	st.balloons()

st.divider()
st.caption("Dibuat oleh **Saskia Humaira**")
