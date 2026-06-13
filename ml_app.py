import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import ml package
import joblib
import os   
        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    # input data
    st.header("Input Data")

    # upload file
    uploaded_file = st.file_uploader("Masukkan File (xlsx)", type="xlsx")

    # display file
    if uploaded_file is not None:
        st.header("Data yang Diunggah")
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        df_cp = df.copy()

        # No 1: Program studi mahasiswa
        prodi_dict = {
            "Program Teknik Informatika": 1,
            "Program Sistem Informasi Bisnis": 2,
            "Program Multimedia": 3
        }

        df_cp["Prodi"] = df_cp["Prodi"].map(prodi_dict)

        # No 2: Peminatan SMU
        peminatan_dict = {
            "IPA": 1,
            "IPS": 2,
            "Bahasa": 3,
            "Lainnya": 4
        }

        df_cp["Peminatan"] = df_cp["Peminatan"].map(peminatan_dict)

        # No 3: Mat
        df_cp["Mat"] = pd.cut(df_cp["Mat"], bins=[-np.inf, 73, 80, 87, 94, np.inf], right=False, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # No 4: Inggris
        df_cp["Inggris"] = pd.cut(df_cp["Inggris"], bins=[-np.inf, 73, 80, 87, 94, np.inf], right=False, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # No 5: RataNilai
        df_cp["RataNilai"] = pd.cut(df_cp["RataNilai"], bins=[-np.inf, 73, 80, 87, 94, np.inf], right=False, labels=[1, 2, 3, 4, 5]).astype(int)

        # No 6: IPKSbl (0 < IPK <= 1, dst. Maka right=True)
        df_cp["IPKSbl"] = pd.cut(df_cp["IPKSbl"], bins=[0, 1, 2, 3, 4], right=True, labels=[1, 2, 3, 4]).astype(int)

        # No 7: SksKumSbl
        df_cp["SksKumSbl"] = pd.cut(df_cp["SksKumSbl"], bins=[0, 20, 40, 60, 80, 100, 120, 140], right=False, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)

        # No 8: JPeserta
        df_cp["JPeserta"] = pd.cut(df_cp["JPeserta"], bins=[1, 23, 45, 67, 89, 111], right=False, labels=[1, 2, 3, 4, 5]).astype(int)

        # No 9: NilaiUlang
        nilai_grade_dict = {
            "E": 0,
            "D": 1,
            "C": 2,
            "BC": 3,
            "B": 4,
            "AB": 5,
            "A": 6,
            "Tidak Ada": 7
        }

        df_cp["NilaiUlang"] = df_cp["NilaiUlang"].map(nilai_grade_dict)

        # No 11: NilaiPrasyarat
        df_cp["NilaiPrasyarat"] = df_cp["NilaiPrasyarat"].map(nilai_grade_dict)

        # No 13: LulusSbl
        df_cp["LulusSbl"] = pd.cut(df_cp["LulusSbl"], bins=[31, 41, 51, 61, 71, 81, 91, 101], right=False, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)
        
        # No 14: UTS (Perhatikan baris terakhir 81 <= UTS <= 100, gunakan inklusif kanan atau sesuaikan batas atas)
        df_cp["UTS"] = pd.cut(df_cp["UTS"], bins=[0, 40, 55, 60, 66, 73, 81, 101], right=False, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)
        
        df_cp["Absen_cat"] = pd.cut(df_cp["Absen"], bins=[-np.inf,0,1,np.inf], labels=[1,2,3]).astype(int)
        df_cp["LamaUlang_cat"] = pd.cut(df_cp["LamaUlang"], bins=[-np.inf,0,1,np.inf], labels=[1,2,3]).astype(int)
        df_cp["LamaPrasyarat_cat"] = pd.cut(df_cp["LamaPrasyarat"], bins=[-np.inf,0,1,3,np.inf], labels=[1,2,3,4]).astype(int)
        
        df_cp.drop(columns=["Nama", "NRP", "Absen", "LamaUlang", "LamaPrasyarat"], inplace=True)

        # hasil prediksi
        st.header("Hasil Prediksi")
        loaded_data = load_model("best_model_with_custom_threshold.pkl")  

        loaded_pipeline = loaded_data['pipeline']
        loaded_threshold = loaded_data['optimal_threshold']

        X_new = df_cp.copy()

        y_pred_proba_new = loaded_pipeline.predict_proba(X_new)[:, 1]
        y_pred_custom_new = (y_pred_proba_new >= loaded_threshold).astype(int)

        df_hasil = df.copy()

        df_hasil["RataNilai"] = df_hasil["RataNilai"].astype(float).map("{:.2f}".format)
        df_hasil["IPKSbl"] = df_hasil["IPKSbl"].astype(float).map("{:.2f}".format)


        df_hasil["Probabilitas Tidak Lulus"] = (y_pred_proba_new * 100).round().astype(int)

        kondisi = [
            (y_pred_proba_new >= loaded_threshold),
            (y_pred_proba_new >= 0.3) & (y_pred_proba_new < loaded_threshold),
            (y_pred_proba_new < 0.3)
        ]

        pilihan_status = [
            "Bahaya, perlu intervensi lebih lanjut",
            "Waspada, perlu dukungan akademik",
            "Aman, perlu dipertahankan"
        ]

        df_hasil["Status Mahasiswa"] = np.select(kondisi, pilihan_status, default="Aman, perlu dipertahankan")

        df_hasil = df_hasil.sort_values(by="Probabilitas Tidak Lulus", ascending=False)
        df_hasil["Probabilitas Tidak Lulus"] = df_hasil["Probabilitas Tidak Lulus"].astype(str) + "%"

        def ganti_warna_status(val):
            if "Bahaya" in val:
                return "background-color: #ffcccc; color: #cc0000; font-weight: bold;" # Merah Lembut
            elif "Waspada" in val:
                return "background-color: #fff2cc; color: #b38600; font-weight: bold;" # Kuning Lembut
            elif "Aman" in val:
                return "background-color: #d9ead3; color: #274e13; font-weight: bold;" # Hijau Lembut
            return ""

        df_final = df_hasil.style.map(ganti_warna_status, subset=["Status Mahasiswa"])

        st.dataframe(df_final) 

        # visualisasi
        st.header("Visualisasi Proporsi Status Mahasiswa")

        status_counts = df_hasil["Status Mahasiswa"].value_counts()

        warna_kamus = {
            "Bahaya, perlu intervensi lebih lanjut": "#ffcccc", # Merah
            "Waspada, perlu dukungan akademik": "#fff2cc",    # Kuning
            "Aman, perlu dipertahankan": "#d9ead3"            # Hijau
        }
        
        colors = [warna_kamus[status] for status in status_counts.index]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            status_counts, 
            labels=status_counts.index, 
            autopct='%1.1f%%',
            startangle=140, 
            colors=colors,
            textprops={'fontsize': 10}
        )
        ax.axis('equal') 

        st.pyplot(fig)