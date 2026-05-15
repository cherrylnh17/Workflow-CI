import argparse
import os
import re
import logging

import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Download NLTK resources
def download_nltk_resources():
    """Download NLTK resources yang diperlukan."""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        nltk.download(resource, quiet=True)
    logger.info("NLTK resources berhasil didownload.")


# Load Data
def load_data(data_path: str, kamusalay_path: str) -> tuple:
    """
    Memuat dataset utama dan kamus normalisasi.

    Args:
        data_path     : Path ke file data.csv
        kamusalay_path: Path ke file new_kamusalay.csv

    Returns:
        df            : DataFrame dataset utama
        kamusalay_dict: Dictionary {slang: formal}
    """
    logger.info(f"Memuat dataset dari: {data_path}")
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")

    logger.info(f"Memuat kamus normalisasi dari: {kamusalay_path}")
    kamusalay_df = pd.read_csv(
        kamusalay_path,
        encoding='ISO-8859-1',
        header=None,
        names=['slang', 'formal']
    )
    kamusalay_dict = dict(zip(kamusalay_df['slang'], kamusalay_df['formal']))
    logger.info(f"Kamus normalisasi dimuat: {len(kamusalay_dict)} entri.")

    return df, kamusalay_dict



# Preprocessing Functions

def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris dengan Tweet kosong dan data duplikat."""
    before = len(df)
    df = df.dropna(subset=['Tweet'])
    df = df.drop_duplicates(subset=['Tweet'])
    df = df.reset_index(drop=True)
    after = len(df)
    logger.info(f"Hapus missing & duplikat: {before} -> {after} baris (-{before - after}).")
    return df


def lowercase_text(text: str) -> str:
    """Mengubah teks menjadi huruf kecil."""
    return str(text).lower()


def remove_noise(text: str) -> str:
    """
    Menghapus elemen noise dari teks:
    - Mention (@user)
    - URL (http/https)
    - Hashtag (#)
    - Angka
    - Tanda baca dan karakter khusus
    - Spasi berlebih
    """
    text = re.sub(r'@[\w]+', 'USER', text)
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'#[\w]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_slang(text: str, kamus: dict) -> str:
    """Menormalisasi kata slang/tidak baku menggunakan kamus kamusalay."""
    words = text.split()
    normalized = [kamus.get(word, word) for word in words]
    return ' '.join(normalized)


def remove_stopwords_id(text: str) -> str:
    """
    Menghapus stopwords Bahasa Indonesia.
    Termasuk kata-kata tidak informatif tambahan.
    """
    stop_words = set(stopwords.words('indonesian'))
    additional_stopwords = {'user', 'url', 'rt', 'yg', 'dgn', 'nya', 'jg', 'aja', 'gak'}
    stop_words.update(additional_stopwords)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(filtered)


def preprocess_text(text: str, kamus: dict) -> str:
    """
    Pipeline preprocessing lengkap:
    1. Lowercase
    2. Remove noise (mention, URL, hashtag, angka, tanda baca)
    3. Normalisasi slang
    4. Stopword removal
    """
    text = lowercase_text(text)
    text = remove_noise(text)
    text = normalize_slang(text, kamus)
    text = remove_stopwords_id(text)
    return text


def apply_preprocessing(df: pd.DataFrame, kamusalay_dict: dict) -> pd.DataFrame:
    """Menerapkan preprocessing pada kolom Tweet."""
    logger.info("Menerapkan preprocessing teks...")
    df['Tweet_clean'] = df['Tweet'].apply(
        lambda x: preprocess_text(x, kamusalay_dict)
    )
    # Hapus baris yang tweetnya kosong setelah preprocessing
    before = len(df)
    df = df[df['Tweet_clean'].str.strip() != ''].reset_index(drop=True)
    after = len(df)
    logger.info(f"Hapus tweet kosong pasca preprocessing: {before} -> {after} baris.")
    return df



# Feature Extraction

def extract_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    max_features: int = 10000
) -> tuple:
    """
    Split data dan ekstraksi fitur TF-IDF.

    Args:
        df          : DataFrame dengan kolom 'Tweet_clean' dan 'HS'
        test_size   : Proporsi data test
        max_features: Jumlah maksimum fitur TF-IDF

    Returns:
        X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer
    """
    X = df['Tweet_clean']
    y = df['HS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test.")

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    logger.info(f"TF-IDF: train shape={X_train_tfidf.shape}, test shape={X_test_tfidf.shape}.")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, X_train, X_test, tfidf



# Save Outputs

def save_outputs(
    df: pd.DataFrame,
    X_train_tfidf, X_test_tfidf,
    y_train, y_test,
    X_train, X_test,
    tfidf,
    output_dir: str
) -> None:
    """
    Menyimpan semua output preprocessing ke output_dir:
    - hate_speech_preprocessed.csv
    - X_train.csv, X_test.csv, y_train.csv, y_test.csv
    - X_train_tfidf.npz, X_test_tfidf.npz
    - tfidf_vectorizer.pkl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dataframe bersih
    label_cols = ['Tweet_clean', 'HS', 'Abusive', 'HS_Weak', 'HS_Moderate', 'HS_Strong',
                  'HS_Individual', 'HS_Group', 'HS_Religion', 'HS_Race',
                  'HS_Physical', 'HS_Gender', 'HS_Other', 'HS_Keyword', 'HS_Repeated']
    existing_cols = [c for c in label_cols if c in df.columns]
    df[existing_cols].to_csv(
        os.path.join(output_dir, 'hate_speech_preprocessed.csv'), index=False
    )

    # Train/test text split
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False, header=['Tweet_clean'])
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False, header=['Tweet_clean'])
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=['HS'])
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=['HS'])

    # TF-IDF matrices (sparse format)
    sp.save_npz(os.path.join(output_dir, 'X_train_tfidf.npz'), X_train_tfidf)
    sp.save_npz(os.path.join(output_dir, 'X_test_tfidf.npz'), X_test_tfidf)

    # TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))

    logger.info(f"Semua output disimpan di: {output_dir}/")
    logger.info("File yang disimpan:")
    for f in sorted(os.listdir(output_dir)):
        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
        logger.info(f"  - {f} ({size_kb:.1f} KB)")



# Main Pipeline

def run_preprocessing_pipeline(
    data_path: str,
    kamusalay_path: str,
    output_dir: str,
    test_size: float = 0.2,
    max_features: int = 10000
) -> None:
    """
    Menjalankan seluruh pipeline preprocessing dari awal sampai akhir.

    Args:
        data_path     : Path ke data.csv
        kamusalay_path: Path ke new_kamusalay.csv
        output_dir    : Direktori output
        test_size     : Proporsi test split
        max_features  : Jumlah fitur TF-IDF
    """
    logger.info("=" * 55)
    logger.info("MULAI PIPELINE PREPROCESSING HATE SPEECH DATASET")
    logger.info("=" * 55)

    # Download NLTK
    download_nltk_resources()

    # Load data
    df, kamusalay_dict = load_data(data_path, kamusalay_path)

    # Bersihkan missing & duplikat
    df = clean_missing_and_duplicates(df)

    # Preprocessing teks
    df = apply_preprocessing(df, kamusalay_dict)

    # Feature extraction & split
    (X_train_tfidf, X_test_tfidf,
     y_train, y_test,
     X_train, X_test, tfidf) = extract_features(df, test_size, max_features)

    # Simpan output
    save_outputs(
        df,
        X_train_tfidf, X_test_tfidf,
        y_train, y_test,
        X_train, X_test,
        tfidf,
        output_dir
    )

    logger.info("=" * 55)
    logger.info("PREPROCESSING SELESAI — DATA SIAP UNTUK PELATIHAN")
    logger.info(f"  Total data final : {len(df)}")
    logger.info(f"  Fitur TF-IDF     : {X_train_tfidf.shape[1]}")
    logger.info(f"  Train samples    : {X_train_tfidf.shape[0]}")
    logger.info(f"  Test samples     : {X_test_tfidf.shape[0]}")
    logger.info(f"  HS label ratio   : {y_train.mean()*100:.1f}% (train)")
    logger.info("=" * 55)



# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing: Indonesian Hate Speech Dataset'
    )
    parser.add_argument(
        '--data', type=str, default='data.csv',
        help='Path ke file data.csv (default: data.csv)'
    )
    parser.add_argument(
        '--kamusalay', type=str, default='new_kamusalay.csv',
        help='Path ke file new_kamusalay.csv (default: new_kamusalay.csv)'
    )
    parser.add_argument(
        '--output', type=str, default='data_preprocessing',
        help='Direktori output (default: data_preprocessing)'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help='Proporsi data test (default: 0.2)'
    )
    parser.add_argument(
        '--max_features', type=int, default=10000,
        help='Jumlah maksimum fitur TF-IDF (default: 10000)'
    )

    args = parser.parse_args()

    run_preprocessing_pipeline(
        data_path=args.data,
        kamusalay_path=args.kamusalay,
        output_dir=args.output,
        test_size=args.test_size,
        max_features=args.max_features
    )
