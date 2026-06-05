import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    INR_TO_USD = 1 / 83.5
    df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    df['selling_price'] = df['selling_price'] * INR_TO_USD
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_filtered = df[df['year'] >= 2016].copy()
    Q1_km = df_filtered['km_driven'].quantile(0.25)
    Q3_km = df_filtered['km_driven'].quantile(0.75)
    IQR_km = Q3_km - Q1_km
    df_clean = df_filtered[
        (df_filtered['km_driven'] >= Q1_km - 1.5*IQR_km) &
        (df_filtered['km_driven'] <= Q3_km + 1.5*IQR_km)
    ].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

model_data = load_model()
model      = model_data['model']
features   = model_data['feature_columns']
r2         = model_data['r2_score']
mape       = model_data['mape']
df         = load_data()

# SIDEBAR
st.sidebar.title("🔧 Input Parameters")
st.sidebar.markdown("Masukkan spesifikasi mobil yang ingin diprediksi harganya.")
year         = st.sidebar.slider("Tahun Produksi", min_value=2016, max_value=2020, value=2018)
km_driven    = st.sidebar.number_input("Kilometer Tempuh", min_value=0, max_value=200000, value=30000, step=1000)
fuel         = st.sidebar.selectbox("Jenis Bahan Bakar", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type  = st.sidebar.selectbox("Tipe Penjual", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmisi", ["Manual", "Automatic"])
owner        = st.sidebar.selectbox("Kepemilikan", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model: Decision Tree (Optimized)**")
st.sidebar.metric("R² Score", f"{r2:.4f}")
st.sidebar.metric("MAPE", f"{mape:.2f}%")

# MAIN
st.title("🚗 Used Car Price Prediction")
st.markdown("Prediksi harga jual mobil bekas menggunakan **Decision Tree (Optimized)** | Kendari Pride")

tab1, tab2, tab3 = st.tabs(["🎯 Prediksi Harga", "📊 Data Exploration", "📈 Model Performance"])

# TAB 1
with tab1:
    st.header("Prediksi Harga Jual Mobil")
    st.markdown("Atur parameter di sidebar kiri, lalu klik tombol prediksi.")

    if st.button("🔮 Prediksi Harga", type="primary"):
        input_dict = {col: 0 for col in features}
        input_dict['year']      = year
        input_dict['km_driven'] = km_driven

        fuel_map = {'Diesel': 'fuel_Diesel', 'LPG': 'fuel_LPG', 'Petrol': 'fuel_Petrol'}
        if fuel in fuel_map and fuel_map[fuel] in input_dict:
            input_dict[fuel_map[fuel]] = 1

        if seller_type == 'Individual' and 'seller_type_Individual' in input_dict:
            input_dict['seller_type_Individual'] = 1
        elif seller_type == 'Trustmark Dealer' and 'seller_type_Trustmark Dealer' in input_dict:
            input_dict['seller_type_Trustmark Dealer'] = 1

        if transmission == 'Manual' and 'transmission_Manual' in input_dict:
            input_dict['transmission_Manual'] = 1

        owner_map = {
            'Second Owner': 'owner_Second Owner',
            'Third Owner': 'owner_Third Owner',
            'Fourth & Above Owner': 'owner_Fourth & Above Owner',
            'Test Drive Car': 'owner_Test Drive Car'
        }
        if owner in owner_map and owner_map[owner] in input_dict:
            input_dict[owner_map[owner]] = 1

        input_df   = pd.DataFrame([input_dict])[features]
        prediction = model.predict(input_df)[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Prediksi Harga (USD)", f"$ {prediction:,.2f}")
        col2.metric("📅 Tahun Produksi", year)
        col3.metric("🛣️ Kilometer", f"{km_driven:,} km")
        st.success(f"Mobil tahun {year} dengan {km_driven:,} km tempuh diprediksi berharga **$ {prediction:,.2f}**")

        avg_price = df[df['year'] == year]['selling_price'].mean()
        if not np.isnan(avg_price):
            diff     = prediction - avg_price
            diff_pct = (diff / avg_price) * 100
            if diff > 0:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di atas** rata-rata pasar tahun {year} ($ {avg_price:,.2f})")
            else:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di bawah** rata-rata pasar tahun {year} ($ {avg_price:,.2f})")

# TAB 2
with tab2:
    st.header("Data Exploration")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Rata-rata Harga", f"$ {df['selling_price'].mean():,.2f}")
    col3.metric("Harga Tertinggi", f"$ {df['selling_price'].max():,.2f}")
    col4.metric("Harga Terendah", f"$ {df['selling_price'].min():,.2f}")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Harga Jual")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['selling_price'], kde=True, color='steelblue', ax=ax)
        ax.set_xlabel("Harga Jual ($)")
        ax.set_ylabel("Frekuensi")
        ax.ticklabel_format(style='plain', axis='x')
        st.pyplot(fig); plt.close()

    with c2:
        st.subheader("Rata-rata Harga per Tahun")
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_year = df.groupby('year')['selling_price'].mean().reset_index()
        ax.plot(avg_year['year'], avg_year['selling_price'], color='seagreen', linewidth=2, marker='o', markersize=6)
        ax.set_xlabel("Tahun Produksi")
        ax.set_ylabel("Rata-rata Harga ($)")
        ax.set_xticks(avg_year['year'].unique())
        ax.ticklabel_format(style='plain', axis='both')
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Distribusi Jenis Bahan Bakar")
        fig, ax = plt.subplots(figsize=(6, 4))
        fuel_counts = df['fuel'].value_counts()
        ax.bar(fuel_counts.index, fuel_counts.values, color='steelblue')
        ax.set_xlabel("Fuel Type")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig); plt.close()

    with c4:
        st.subheader("Harga per Jenis Transmisi")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='transmission', y='selling_price', data=df, palette='Set2', ax=ax)
        ax.set_xlabel("Transmisi")
        ax.set_ylabel("Harga Jual ($)")
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig); plt.close()

# TAB 3
with tab3:
    st.header("Model Performance")
    st.markdown("Model yang digunakan: **Decision Tree (Optimized)** dengan `max_depth=5`")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAPE", f"{mape:.2f}%")

    verdict = "Highly Accurate" if mape < 10 else "Good" if mape < 20 else "Reasonable"
    st.success(f"Model Verdict: **{verdict}**")

    st.markdown("---")
    st.subheader("Perbandingan Model")
    st.markdown("""
| Model | Keterangan |
|-------|-----------|
| Linear Regression | Baseline, paling sederhana |
| Decision Tree (Optimized) | **Model terpilih**, max_depth=5 |
| Random Forest | Ensemble, lebih kompleks |
""")

    st.markdown("---")
    st.subheader("Best Parameters")
    st.dataframe(pd.DataFrame({
        'Parameter': ['max_depth', 'random_state'],
        'Value': [5, 42]
    }), use_container_width=True)

st.markdown("---")
st.markdown("**ISB02303402 Data Science** | Capstone Project 2025/2026 | **Kendari Pride**")
