import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    INR_TO_USD = 0.012
    df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    df['selling_price'] = df['selling_price'] * INR_TO_USD
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df['year'] >= 2014].copy()
    df.reset_index(drop=True, inplace=True)
    # Outlier removal IQR
    Q1 = df['selling_price'].quantile(0.25)
    Q3 = df['selling_price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['selling_price'] >= Q1 - 1.5*IQR) & (df['selling_price'] <= Q3 + 1.5*IQR)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

model_data = load_model()
model      = model_data['model']
features   = model_data['feature_columns']
r2         = model_data['r2_score']
mape       = model_data['mape']
df         = load_data()

# ============================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================
st.sidebar.title("🔧 Input Parameters")
st.sidebar.markdown("Masukkan spesifikasi mobil yang ingin diprediksi harganya.")

year         = st.sidebar.slider("Tahun Produksi", min_value=2014, max_value=2020, value=2017)
km_driven    = st.sidebar.number_input("Kilometer Tempuh", min_value=0, max_value=500000, value=50000, step=1000)
fuel         = st.sidebar.selectbox("Jenis Bahan Bakar", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type  = st.sidebar.selectbox("Tipe Penjual", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmisi", ["Manual", "Automatic"])
owner        = st.sidebar.selectbox("Kepemilikan", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.metric("R² Score", f"{r2:.4f}")
st.sidebar.metric("MAPE", f"{mape:.2f}%")

# ============================================================
# MAIN - TABS
# ============================================================
st.title("🚗 Used Car Price Prediction")
st.markdown("Aplikasi prediksi harga jual mobil bekas menggunakan Random Forest Regressor")

tab1, tab2, tab3 = st.tabs(["🎯 Prediksi Harga", "📊 Data Exploration", "📈 Model Performance"])

# ============================================================
# TAB 1: PREDIKSI
# ============================================================
with tab1:
    st.header("Prediksi Harga Jual Mobil")
    st.markdown("Atur parameter mobil di sidebar kiri, kemudian klik tombol prediksi.")

    if st.button("🔮 Prediksi Harga", type="primary"):

        # Build input dataframe sesuai feature columns
        owner_map = {
            "First Owner": 0,
            "Second Owner": 1,
            "Third Owner": 2,
            "Fourth & Above Owner": 3,
            "Test Drive Car": 4
        }

        input_dict = {
            'year': year,
            'km_driven': km_driven,
            'owner': owner_map[owner],
            'fuel_CNG': 1 if fuel == 'CNG' else 0,
            'fuel_Diesel': 1 if fuel == 'Diesel' else 0,
            'fuel_LPG': 1 if fuel == 'LPG' else 0,
            'fuel_Petrol': 0,  # drop_first reference
            'seller_type_Individual': 1 if seller_type == 'Individual' else 0,
            'seller_type_Trustmark Dealer': 1 if seller_type == 'Trustmark Dealer' else 0,
            'transmission_Manual': 1 if transmission == 'Manual' else 0,
        }

        # Buat dataframe dengan kolom sesuai training
        input_df = pd.DataFrame([input_dict])

        # Pastikan kolom sesuai dengan features training
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]

        # Prediksi
        prediction = model.predict(input_df)[0]

        # Tampilkan hasil
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Prediksi Harga (USD)", f"$ {prediction:,.2f}")
        with col2:
            st.metric("📅 Tahun Produksi", year)
        with col3:
            st.metric("🛣️ Kilometer", f"{km_driven:,} km")

        st.success(f"Mobil {year} dengan {km_driven:,} km tempuh diprediksi berharga **$ {prediction:,.2f}**")

        # Bandingkan dengan rata-rata pasar
        avg_price = df[df['year'] == year]['selling_price'].mean()
        if not np.isnan(avg_price):
            diff = prediction - avg_price
            diff_pct = (diff / avg_price) * 100
            if diff > 0:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di atas** rata-rata pasar mobil tahun {year} ($ {avg_price:,.2f})")
            else:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di bawah** rata-rata pasar mobil tahun {year} ($ {avg_price:,.2f})")

# ============================================================
# TAB 2: DATA EXPLORATION
# ============================================================
with tab2:
    st.header("Data Exploration")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Rata-rata Harga", f"$ {df['selling_price'].mean():,.2f}")
    col3.metric("Harga Tertinggi", f"$ {df['selling_price'].max():,.2f}")
    col4.metric("Harga Terendah", f"$ {df['selling_price'].min():,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Harga Jual")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['selling_price'], kde=True, color='steelblue', ax=ax)
        ax.set_xlabel("Harga Jual ($)")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Rata-rata Harga per Tahun")
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_year = df.groupby('year')['selling_price'].mean().reset_index()
        ax.plot(avg_year['year'], avg_year['selling_price'], color='seagreen', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Tahun Produksi")
        ax.set_ylabel("Rata-rata Harga ($)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Jenis Bahan Bakar")
        fig, ax = plt.subplots(figsize=(6, 4))
        fuel_counts = df['fuel'].value_counts()
        ax.bar(fuel_counts.index, fuel_counts.values, color='steelblue')
        ax.set_xlabel("Fuel Type")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Harga per Jenis Transmisi")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='transmission', y='selling_price', data=df, palette='Set2', ax=ax)
        ax.set_xlabel("Transmisi")
        ax.set_ylabel("Harga Jual ($)")
        st.pyplot(fig)
        plt.close()

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.header("Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("R² Score (Random Forest Tuned)", f"{r2:.4f}")
    col2.metric("MAPE", f"{mape:.2f}%")

    verdict = "Highly Accurate" if mape < 10 else "Good" if mape < 20 else "Reasonable"
    st.success(f"Model Verdict: **{verdict}**")

    st.markdown("---")
    st.subheader("Feature Importance")

    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_imp, x='Importance', y='Feature', color='steelblue', ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Best Parameters (GridSearchCV)")
    params_df = pd.DataFrame(
        model_data['best_params'].items(),
        columns=['Parameter', 'Value']
    )
    st.dataframe(params_df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("**ISB02303402 Data Science** | Capstone Project 2025/2026 | Kendari Pride")
