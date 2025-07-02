# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords (first time only)
nltk.download('stopwords')

# ===============================
# LOAD AND CLEAN THE EXCEL FILE
# ===============================
file_path = 'SupplementaryTable1.xlsx'  # Excel file given by Siddhartha sir
df = pd.read_excel(file_path)

# Skip header row and reset index
df = df.iloc[1:].reset_index(drop=True)

# Rename columns for clarity
df = df.rename(columns={
    'MDPI': 'SlNo_MDPI',
    'Unnamed: 5': 'Keywords_MDPI',
    'Unnamed: 6': 'Abstract_MDPI',
    'Unnamed: 25': 'Keywords_SciDir',
    'Unnamed: 24': 'Abstract_SciDir'
})

# Clean missing values and combine keyword fields
df['Keywords_MDPI'] = df['Keywords_MDPI'].astype(str).fillna('')
df['Keywords_SciDir'] = df['Keywords_SciDir'].astype(str).fillna('')
df['All_Keywords'] = df['Keywords_MDPI'] + '; ' + df['Keywords_SciDir']

# Function to extract cleaned keyword list
def extract_keywords(text):
    return [kw.strip().lower() for kw in text.split(';') if kw.strip()]

df['Keyword_List'] = df['All_Keywords'].apply(extract_keywords)

# Flatten all keywords
all_keywords = [kw for group in df['Keyword_List'] for kw in group]
keyword_counter = Counter(all_keywords)

# Get common and unique keywords
common_keywords = [kw for kw, count in keyword_counter.items() if count > 1]
unique_keywords = [kw for kw, count in keyword_counter.items() if count == 1]

# ===============================
# SEARCH FUNCTION
# ===============================
def search_articles(keyword):
    keyword = keyword.lower().strip()
    results = df[df['Keyword_List'].apply(lambda kws: keyword in kws)]
    return results[['SlNo_MDPI', 'All_Keywords', 'Abstract_MDPI', 'Abstract_SciDir']]

# ===============================
# SEMANTIC SIMILARITY FUNCTION
# ===============================
def find_similar_keywords(threshold=0.75):
    filtered = list(set([
        kw for kw in all_keywords
        if kw not in stopwords.words('english') and len(kw) > 2
    ]))

    vectorizer = TfidfVectorizer().fit(filtered)
    vectors = vectorizer.transform(filtered)
    sim_matrix = cosine_similarity(vectors)

    similar_pairs = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            if sim_matrix[i, j] >= threshold:
                similar_pairs.append((filtered[i], filtered[j], round(sim_matrix[i, j], 2)))
    return similar_pairs

# ===============================
# USER INTERFACE
# ===============================
if __name__ == "__main__":
    print("🔍 Welcome! This tool helps you analyze article keywords.\n")

    try:
        n_common = int(input("How many common keywords would you like to see? "))
        n_unique = int(input("How many unique keywords would you like to see? "))
        n_similar = int(input("How many similar keyword pairs would you like to see? "))
    except ValueError:
        print("❌ Please enter valid integer values.")
        exit()

    print("\n🔹 Top Common Keywords:")
    print(common_keywords[:n_common] if n_common <= len(common_keywords) else common_keywords)

    print("\n🔹 Top Unique Keywords:")
    print(unique_keywords[:n_unique] if n_unique <= len(unique_keywords) else unique_keywords)

    print("\n🔁 Finding Similar Keywords (This may take a few seconds)...")
    similar_keywords = find_similar_keywords(threshold=0.8)

    print("\n🔹 Top Similar Keyword Pairs (Cosine similarity ≥ 0.8):")
    for kw1, kw2, sim in similar_keywords[:n_similar]:
        print(f"  {kw1} ⟷ {kw2}  (Similarity: {sim})")

    # Optional keyword search
    choice = input("\nDo you want to search for articles using a keyword? (y/n): ").strip().lower()
    if choice == 'y':
        search_kw = input("Enter keyword to search: ")
        results = search_articles(search_kw)
        if results.empty:
            print("❌ No articles found with that keyword.")
        else:
            print(f"\n🔍 Articles with keyword '{search_kw}':")
            print(results.to_string(index=False))
