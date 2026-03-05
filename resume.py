import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy for Natural Language Processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If the model is not found, download it automatically
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Basic cleaning of text for better matching."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_keywords(text):
    """Extracts nouns and proper nouns as 'skills' using NLP."""
    doc = nlp(clean_text(text))
    return set([token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])

def main():
    # 1. Load the Dataset
    try:
        df = pd.read_csv('resume_data.csv')
        # Clean column names to remove hidden characters (like \ufeff)
        df.columns = df.columns.str.strip().str.replace('﻿', '')
    except FileNotFoundError:
        print("Error: 'resume_data.csv' not found in the current folder.")
        return

    print("\n--- 🤖 AI Resume Screening & Ranking System ---")
    
    # 2. Get Job Role & Fetch Requirements from Dataset
    target_role = input("Enter the Job Role to screen for: ").strip()
    
    # Find the role in your CSV (Job Position Name column)
    role_match = df[df['job_position_name'].str.contains(target_role, case=False, na=False)]
    
    if role_match.empty:
        print(f"⚠️ No requirements found for '{target_role}' in dataset. Using general keyword matching.")
        benchmark_skills = {"python", "data", "analysis", "communication", "management", "technical"}
        benchmark_text = " ".join(list(benchmark_skills))
    else:
        # We take the first match found in the dataset as the 'Gold Standard'
        sample = role_match.iloc[0]
        # Combining the 'skills_required' and 'responsibilities.1' columns for a full requirement profile
        benchmark_text = str(sample.get('skills_required', '')) + " " + str(sample.get('responsibilities.1', ''))
        benchmark_skills = extract_keywords(benchmark_text)
        print(f"✅ Found requirements for '{target_role}' in dataset.")

    # 3. Get Student Resumes at Runtime
    try:
        num_students = int(input("\nHow many student resumes do you want to check? "))
    except ValueError:
        print("Please enter a valid number.")
        return

    candidate_data = []
    for i in range(num_students):
        print(f"\n--- Entry {i+1} ---")
        name = input("Candidate Name: ")
        resume_text = input(f"Paste Resume text for {name}: ")
        candidate_data.append({'Name': name, 'Raw_Resume': resume_text})

    # 4. Analyze and Rank
    results = []
    for person in candidate_data:
        cleaned_res = clean_text(person['Raw_Resume'])
        resume_skills = extract_keywords(cleaned_res)
        
        # Calculate TF-IDF Score
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([clean_text(benchmark_text), cleaned_res])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        
        # Identify Skill Gap
        missing = benchmark_skills - resume_skills
        
        results.append({
            'Rank': 0, # Placeholder
            'Candidate Name': person['Name'],
            'Match Score (%)': round(score, 2),
            'Missing Skills': ", ".join(list(missing)[:8]) if missing else "None (Matched All)"
        })

    # 5. Output Results in Table Format
    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by='Match Score (%)', ascending=False).reset_index(drop=True)
    final_df['Rank'] = final_df.index + 1
    
    print("\n" + "="*70)
    print(f"RANKING RESULTS FOR ROLE: {target_role.upper()}")
    print("="*70)
    print(final_df[['Rank', 'Candidate Name', 'Match Score (%)', 'Missing Skills']].to_string(index=False))

if __name__ == "__main__":
    main()