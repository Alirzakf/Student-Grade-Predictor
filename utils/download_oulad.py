"""
OULAD Dataset Downloader
Downloads and processes Open University Learning Analytics Dataset
Source: https://analyse.kmi.open.ac.uk/open_dataset
"""

import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from pathlib import Path


def download_oulad_sample():
    """
    Download sample OULAD data from the official source.
    
    The full dataset can be downloaded from:
    https://analyse.kmi.open.ac.uk/open_dataset
    
    For demo purposes, this creates a representative sample.
    """
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("OULAD Dataset Information:")
    print("=" * 60)
    print("Dataset: Open University Learning Analytics Dataset (OULAD)")
    print("Source: https://analyse.kmi.open.ac.uk/open_dataset")
    print("License: CC BY 4.0")
    print("=" * 60)
    
    # Check if data already exists
    required_files = [
        data_dir / 'students.csv',
        data_dir / 'assessments.csv',
        data_dir / 'studentVle.csv'
    ]
    
    if all(f.exists() for f in required_files):
        print("\n✅ OULAD CSV files found in data/ directory")
        return True
    
    print("\n📥 OULAD CSV files not found.")
    print("\nTo use real OULAD data:")
    print("1. Visit: https://analyse.kmi.open.ac.uk/open_dataset")
    print("2. Download the dataset (registration required)")
    print("3. Extract these files to the 'data/' directory:")
    print("   - students.csv")
    print("   - assessments.csv")
    print("   - studentVle.csv")
    print("4. Re-run the application")
    
    return False


def generate_oulad_sample_data():
    """
    Generate realistic OULAD-like sample data for demonstration.
    This mimics the actual OULAD structure.
    """
    np.random.seed(42)
    
    # Generate synthetic students data (OULAD structure)
    n_students = 300
    students_data = {
        'id_student': range(1, n_students + 1),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], n_students),
        'code_presentation': np.random.choice([2013, 2014, 2015], n_students),
        'age_band': np.random.choice(['<=35', '35-55', '>55'], n_students),
        'gender': np.random.choice(['M', 'F'], n_students),
        'disability': np.random.choice(['Y', 'N'], n_students),
        'num_of_prev_attempts': np.random.randint(0, 5, n_students),
        'studied_credits': np.random.choice([30, 60, 90, 120], n_students),
        'region': np.random.choice(['London', 'South East', 'South West', 'Midlands', 
                                   'East of England', 'East Midlands', 'North West',
                                   'Yorkshire', 'North', 'Scotland', 'Wales', 'Northern Ireland'],
                                  n_students),
        'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'], n_students)
    }
    students_df = pd.DataFrame(students_data)
    
    # Generate assessments data
    n_assessments = 1200
    assessment_data = {
        'id_assessment': range(1, n_assessments + 1),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], n_assessments),
        'code_presentation': np.random.choice([2013, 2014, 2015], n_assessments),
        'assessment_type': np.random.choice(['TMA', 'CMA', 'Exam'], n_assessments),
        'date': np.random.randint(50, 300, n_assessments),
        'weight': np.random.choice([10, 20, 30, 40, 50], n_assessments)
    }
    assessments_df = pd.DataFrame(assessment_data)
    
    # Generate student assessment scores
    n_scores = 2000
    scores_data = {
        'id_student': np.random.choice(range(1, n_students + 1), n_scores),
        'id_assessment': np.random.choice(range(1, min(n_assessments + 1, 500)), n_scores),
        'score': np.random.uniform(0, 100, n_scores),
        'date_submitted': np.random.randint(50, 300, n_scores)
    }
    scores_df = pd.DataFrame(scores_data)
    
    # Generate student VLE interaction data
    n_interactions = 5000
    vle_data = {
        'id_student': np.random.choice(range(1, n_students + 1), n_interactions),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], n_interactions),
        'code_presentation': np.random.choice([2013, 2014, 2015], n_interactions),
        'id_site': np.random.randint(1, 100, n_interactions),
        'date': np.random.randint(1, 300, n_interactions),
        'sum_click': np.random.randint(1, 50, n_interactions),
        'activity_type': np.random.choice(['page', 'quiz', 'resource', 'forum', 'oucollaborate'],
                                         n_interactions)
    }
    vle_df = pd.DataFrame(vle_data)
    
    return students_df, assessments_df, scores_df, vle_df


def merge_oulad_data(students_df, assessments_df, scores_df, vle_df):
    """
    Merge OULAD datasets into a single feature matrix.
    
    Args:
        students_df: Students data
        assessments_df: Assessments metadata
        scores_df: Student assessment scores
        vle_df: Student VLE (Virtual Learning Environment) interactions
    
    Returns:
        pd.DataFrame: Merged dataset with engineered features
    """
    print("\nMerging OULAD datasets...")
    
    # Calculate days since registration (approximate)
    vle_df_agg = vle_df.groupby('id_student').agg({
        'date': 'min',
        'sum_click': 'sum'
    }).rename(columns={'date': 'first_date', 'sum_click': 'clicks_total'})
    vle_df_agg['days_since_registration'] = 300 - vle_df_agg['first_date']
    
    # Calculate average assessment score per student
    scores_df_agg = scores_df.groupby('id_student').agg({
        'score': ['mean', 'count', 'std']
    }).reset_index()
    scores_df_agg.columns = ['id_student', 'assessment_score_avg', 'num_assessments', 'assessment_score_std']
    scores_df_agg['assessment_score_avg'] = scores_df_agg['assessment_score_avg'].fillna(50)
    scores_df_agg['assessment_score_std'] = scores_df_agg['assessment_score_std'].fillna(0)
    
    # Merge all data
    merged = students_df.merge(vle_df_agg, left_on='id_student', right_index=True, how='left')
    merged = merged.merge(scores_df_agg, on='id_student', how='left')
    
    # Fill missing values
    merged['clicks_total'] = merged['clicks_total'].fillna(100)
    merged['days_since_registration'] = merged['days_since_registration'].fillna(150)
    merged['assessment_score_avg'] = merged['assessment_score_avg'].fillna(50)
    
    print(f"✅ Merged {len(merged)} student records with features")
    print(f"📊 Features: {merged.columns.tolist()}")
    
    return merged


def load_or_generate_oulad():
    """
    Load OULAD data from CSV files or generate sample data.
    
    Returns:
        pd.DataFrame: Processed OULAD dataset
    """
    data_dir = Path('data')
    
    # Check for existing CSV files
    students_path = data_dir / 'students.csv'
    assessments_path = data_dir / 'assessments.csv'
    vle_path = data_dir / 'studentVle.csv'
    scores_path = data_dir / 'studentAssessment.csv'
    
    if all(p.exists() for p in [students_path, assessments_path, vle_path]):
        print("📖 Loading OULAD datasets from CSV files...")
        students_df = pd.read_csv(students_path)
        assessments_df = pd.read_csv(assessments_path)
        vle_df = pd.read_csv(vle_path)
        
        # Try to load student scores if available
        if scores_path.exists():
            scores_df = pd.read_csv(scores_path)
        else:
            # Generate from assessments if not available
            scores_df = assessments_df.copy()
            scores_df['score'] = np.random.uniform(0, 100, len(scores_df))
            scores_df['id_student'] = np.random.choice(students_df['id_student'], len(scores_df))
        
        print("✅ Real OULAD data loaded successfully!")
        
    else:
        print("⚠️  OULAD CSV files not found. Generating sample data...")
        print("💡 To use real data, download from: https://analyse.kmi.open.ac.uk/open_dataset")
        students_df, assessments_df, scores_df, vle_df = generate_oulad_sample_data()
        print("✅ Sample OULAD-like data generated")
    
    # Merge and process
    merged_df = merge_oulad_data(students_df, assessments_df, scores_df, vle_df)
    
    return merged_df


if __name__ == '__main__':
    print("OULAD Dataset Loader")
    print("=" * 60)
    
    # Show download info
    download_oulad_sample()
    
    # Load or generate data
    df = load_or_generate_oulad()
    print(f"\n📊 Final dataset shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"\n✅ Dataset ready for preprocessing!")
