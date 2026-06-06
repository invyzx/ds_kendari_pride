import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="wide")

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_raw_data():
    df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    return df

@st.cache_data
def load_clean_data():
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
df_raw     = load_raw_data()
df         = load_clean_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🚗 Used Car Price Prediction")
st.sidebar.markdown("**Machine Learning Project**")
st.sidebar.markdown("**Model:** Decision Tree (Optimized)")
st.sidebar.markdown("**Dataset:** CarDekho Vehicle Dataset")
st.sidebar.markdown("---")
st.sidebar.markdown("**Input Parameters**")

year         = st.sidebar.slider("Tahun Produksi", min_value=2016, max_value=2020, value=2018)
km_driven    = st.sidebar.number_input("Kilometer Tempuh", min_value=0, max_value=200000, value=30000, step=1000)
fuel         = st.sidebar.selectbox("Jenis Bahan Bakar", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type  = st.sidebar.selectbox("Tipe Penjual", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmisi", ["Manual", "Automatic"])
owner        = st.sidebar.selectbox("Kepemilikan", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

st.sidebar.markdown("---")
st.sidebar.metric("R² Score", f"{r2:.4f}")
st.sidebar.metric("MAPE", f"{mape:.2f}%")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🚗 Used Car Price Prediction")
st.markdown("Prediksi harga jual mobil bekas menggunakan **Decision Tree (Optimized)** | **Kendari Pride**")

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Total Data", f"{len(df):,}")
c2.metric("💰 Rata-rata Harga", f"$ {df['selling_price'].mean():,.2f}")
c3.metric("📈 R² Score", f"{r2:.4f}")
c4.metric("🎯 MAPE", f"{mape:.2f}%")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Prediction",
    "📊 Preliminary Analysis",
    "🔗 Correlation Analysis",
    "⚙️ Model Performance",
    "📋 Dataset",
    "ℹ️ About"
])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab1:
    st.header("Prediksi Harga Jual Mobil")
    st.markdown("Atur parameter mobil di **sidebar kiri**, kemudian klik tombol prediksi.")

    # Show current input summary
    st.info(f"**Input saat ini:** {year} | {km_driven:,} km | {fuel} | {seller_type} | {transmission} | {owner}")

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

        st.markdown("### Hasil Prediksi")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Prediksi Harga (USD)", f"$ {prediction:,.2f}")
        col2.metric("📅 Tahun Produksi", year)
        col3.metric("🛣️ Kilometer", f"{km_driven:,} km")

        st.success(f"Mobil tahun {year} dengan {km_driven:,} km diprediksi berharga **$ {prediction:,.2f}**")

        avg_price = df[df['year'] == year]['selling_price'].mean()
        if not np.isnan(avg_price):
            diff     = prediction - avg_price
            diff_pct = (diff / avg_price) * 100
            if diff > 0:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di atas** rata-rata pasar tahun {year} ($ {avg_price:,.2f})")
            else:
                st.info(f"Harga prediksi **$ {abs(diff):,.2f} ({abs(diff_pct):.1f}%) di bawah** rata-rata pasar tahun {year} ($ {avg_price:,.2f})")

        # Prediction history
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        st.session_state.predictions.append({
            'Tahun': year, 'KM': km_driven, 'Fuel': fuel,
            'Transmisi': transmission, 'Owner': owner,
            'Prediksi ($)': f"$ {prediction:,.2f}"
        })

    # Show prediction history
    if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
        st.markdown("### Riwayat Prediksi")
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)

# ============================================================
# TAB 2: PRELIMINARY ANALYSIS (semua EDA dari notebook)
# ============================================================
with tab2:
    st.header("Preliminary Analysis")
    st.markdown("Eksplorasi data setelah cleaning (filter tahun ≥ 2016, outlier removal km_driven)")

    # 1. Distribusi harga jual
    st.subheader("Distribusi Harga Jual (USD)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['selling_price'], kde=True, color='steelblue', ax=ax)
    ax.set_xlabel("Harga Jual ($)")
    ax.set_ylabel("Frekuensi")
    ax.ticklabel_format(style='plain', axis='x')
    st.pyplot(fig); plt.close()
    st.markdown("Distribusi harga jual bersifat **right-skewed**, mayoritas mobil berada di range harga menengah ke bawah.")
    st.markdown("---")

    # 2. Distribusi fuel type
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Jenis Bahan Bakar")
        fig, ax = plt.subplots(figsize=(6, 4))
        fuel_counts = df['fuel'].value_counts()
        ax.bar(fuel_counts.index, fuel_counts.values, color='steelblue')
        ax.set_xlabel("Fuel Type")
        ax.set_ylabel("Jumlah")
        total = len(df)
        for j, (idx, val) in enumerate(zip(fuel_counts.index, fuel_counts.values)):
            ax.text(j, val, f"{val/total*100:.1f}%", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Harga per Jenis Transmisi")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='transmission', y='selling_price', data=df, palette='Set2', ax=ax)
        ax.set_xlabel("Transmisi")
        ax.set_ylabel("Harga Jual ($)")
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig); plt.close()

    st.markdown("Listing didominasi **Petrol** dan **Diesel**. Mobil **Automatic** rata-rata lebih mahal dari Manual.")
    st.markdown("---")

    # 3. Boxplot numerik
    st.subheader("Boxplot Fitur Numerik")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cols = ['selling_price', 'km_driven', 'year']
    colors = ['steelblue', 'darkorange', 'seagreen']
    labels = ['Harga Jual ($)', 'Kilometer Tempuh', 'Tahun Produksi']
    for i, (col, color, label) in enumerate(zip(cols, colors, labels)):
        sns.boxplot(y=df[col], ax=axes[i], color=color)
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].set_ylabel(label)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("Setelah outlier removal, range data lebih wajar. Tahun produksi terdistribusi dari 2016 hingga 2020.")
    st.markdown("---")

    # 4. Line chart year vs price dan km vs price
    st.subheader("Tren Harga Jual")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_year = df.groupby('year')['selling_price'].mean().reset_index()
        ax.plot(avg_year['year'], avg_year['selling_price'], color='seagreen', linewidth=2, marker='o', markersize=6)
        ax.set_title("Rata-rata Harga per Tahun")
        ax.set_xlabel("Tahun Produksi")
        ax.set_ylabel("Rata-rata Harga ($)")
        ax.set_xticks(avg_year['year'].unique())
        ax.ticklabel_format(style='plain', axis='both')
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_temp = df.copy()
        df_temp['km_bin'] = pd.cut(df_temp['km_driven'], bins=10)
        avg_km = df_temp.groupby('km_bin')['selling_price'].mean().reset_index()
        ax.plot(range(len(avg_km)), avg_km['selling_price'], color='darkorange', linewidth=2, marker='o', markersize=6)
        ax.set_title("Rata-rata Harga per Range Kilometer")
        ax.set_xlabel("Range KM (rendah ke tinggi)")
        ax.set_ylabel("Rata-rata Harga ($)")
        ax.ticklabel_format(style='plain', axis='y')
        st.pyplot(fig); plt.close()

    st.markdown("Mobil tahun lebih baru cenderung lebih mahal. Semakin tinggi kilometer, harga cenderung turun.")

# ============================================================
# TAB 3: CORRELATION ANALYSIS
# ============================================================
with tab3:
    st.header("Correlation Analysis")

    st.subheader("Correlation Heatmap")
    numeric_cols = ['selling_price', 'km_driven', 'year']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap='RdBu', annot=True, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig); plt.close()

    st.markdown("""
**Interpretasi:**
- `year` berkorelasi **positif** dengan `selling_price`: mobil baru = harga lebih tinggi
- `km_driven` berkorelasi **negatif** dengan `selling_price`: km tinggi = harga lebih rendah
    """)

    st.markdown("---")
    st.subheader("Korelasi terhadap Selling Price")
    corr_target = corr['selling_price'].drop('selling_price').sort_values(key=abs, ascending=False)
    st.dataframe(corr_target.reset_index().rename(columns={'index': 'Feature', 'selling_price': 'Correlation'}), use_container_width=True)

# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================
with tab4:
    st.header("Model Performance")

    # Model comparison table
    st.subheader("Model Comparison")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df_model = pd.get_dummies(
        df.drop(columns=['name']),
        columns=['fuel', 'seller_type', 'transmission', 'owner'],
        drop_first=True
    )
    bool_cols_m = df_model.select_dtypes(include='bool').columns
    df_model[bool_cols_m] = df_model[bool_cols_m].astype(int)

    X = df_model.drop(columns=['selling_price'])
    y = df_model['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_list = {
        'Linear Regression': LinearRegression(),
        'Decision Tree (max_depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    results = []
    preds = {}
    for name, m in models_list.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        preds[name] = pred
        r2_val = r2_score(y_test, pred)
        rmse_val = np.sqrt(mean_squared_error(y_test, pred))
        mape_val = np.mean(np.abs((y_test - pred) / y_test)) * 100
        results.append({'Model': name, 'R² Score': round(r2_val, 4), 'RMSE ($)': round(rmse_val, 2), 'MAPE (%)': round(mape_val, 2)})

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
    st.success(f"🏆 Best Model: **{best_model_name}**")

    # Bar chart comparison
    st.subheader("R² Score Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(results_df['Model'], results_df['R² Score'], color='steelblue')
    ax.set_ylabel("R² Score")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, results_df['R² Score']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # Scatter actual vs predicted
    st.subheader("Actual vs Predicted Price")
    best_pred = preds[best_model_name]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, best_pred, alpha=0.4, color='steelblue', label='Prediksi')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Actual vs Predicted ({best_model_name})")
    ax.ticklabel_format(style='plain', axis='both')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.markdown("Titik yang dekat garis merah = prediksi akurat. Titik yang jauh = error tinggi.")

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance")
    best_m = models_list[best_model_name]
    if hasattr(best_m, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_m.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=feat_imp, x='Importance', y='Feature', color='steelblue', ax=ax)
        ax.set_title(f"Feature Importance - {best_model_name}")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ============================================================
# TAB 5: DATASET
# ============================================================
with tab5:
    st.header("Dataset")

    st.subheader("Dataset Preview (Raw)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Statistical Summary (After Cleaning)")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values (Raw)")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing']
    st.dataframe(missing, use_container_width=True)

    st.markdown("---")

    st.subheader("Data Cleaning Summary")
    st.markdown(f"""
| Step | Keterangan | Jumlah |
|------|-----------|--------|
| Data awal | Total records | {len(df_raw):,} |
| Drop duplicates | Duplikat dihapus | {df_raw.duplicated().sum():,} |
| Filter tahun | Hanya ≥ 2016 | - |
| Outlier removal | IQR pada km_driven | - |
| Data final | Siap untuk modeling | {len(df):,} |
    """)

# ============================================================
# TAB 6: ABOUT
# ============================================================
with tab6:
    st.header("About Project")

    st.subheader("Used Car Price Prediction")
    st.markdown("Tujuan proyek ini adalah memprediksi harga jual mobil bekas menggunakan Machine Learning berdasarkan metodologi **CRISP-DM**.")

    st.markdown("---")

    st.subheader("CRISP-DM Methodology")
    st.markdown("""
| Fase | Keterangan |
|------|-----------|
| Business Understanding | Memahami kebutuhan prediksi harga mobil bekas untuk dealer dan pembeli |
| Data Understanding | Eksplorasi dataset CarDekho, identifikasi pola dan distribusi |
| Data Preparation | Cleaning, filtering tahun ≥ 2016, outlier removal, encoding |
| Modeling | Membandingkan Linear Regression, Decision Tree, dan Random Forest |
| Evaluation | Evaluasi menggunakan R², RMSE, dan MAPE |
| Deployment | Deploy aplikasi prediksi ke Streamlit Community Cloud |
    """)

    st.markdown("---")

    st.subheader("Dataset")
    st.markdown("**CAR DETAILS FROM CAR DEKHO** dari Kaggle")
    st.markdown("https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho")

    st.markdown("---")

    st.subheader("Models")
    st.markdown(f"""
| Model | R² Score | MAPE |
|-------|---------|------|
| Linear Regression | {results_df.iloc[0]['R² Score']:.4f} | {results_df.iloc[0]['MAPE (%)']:.2f}% |
| Decision Tree (Optimized) | {results_df.iloc[1]['R² Score']:.4f} | {results_df.iloc[1]['MAPE (%)']:.2f}% |
| Random Forest | {results_df.iloc[2]['R² Score']:.4f} | {results_df.iloc[2]['MAPE (%)']:.2f}% |
    """)

    st.markdown("---")

    st.subheader("Target Variable")
    st.markdown("- `selling_price` (USD) = harga jual mobil bekas")
    st.markdown("- Konversi: 1 INR = 1/83.5 USD")

    st.markdown("---")

    st.subheader("Team - Kendari Pride")
    st.markdown("""
| Nama | NIM |
|------|-----|
| Ekkin Kenneth Hosari | 0706022410023 |
| Marcello Richie Wijaya | 0706022410037 |
| Rayyan Hasan Bafadal | 0706022410011 |
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("**🚗 Used Car Price Prediction**")
    st.markdown("ISB02303402 Data Science | Capstone Project 2025/2026")
    st.markdown("**Kendari Pride**")
