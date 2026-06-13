import streamlit as st

from ml_app import run_ml_app

def main():    
    st.title("Identifikasi Mahasiswa Berpotensi Tidak Lulus", text_alignment="center")
    run_ml_app()

if __name__ == '__main__':
    main()