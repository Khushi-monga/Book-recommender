import pandas as pd
import ast
from collections import Counter

df = pd.read_csv('cleaned_books.csv')
print(f"BEFORE: {len(df):,} books")

# ── STEP 1: Parse genre strings → lists ─────────────────────────────────────
def parse_genres(g):
    try: return ast.literal_eval(g) if isinstance(g, str) else []
    except: return []

df['genres_list'] = df['genres'].apply(parse_genres)

# ── STEP 2: Genre merge map (789 raw tags → 20 clean genres) ────────────────
GENRE_MAP = {
    'fiction': 'Fiction', 'literary fiction': 'Fiction', 'adult fiction': 'Fiction',
    'realistic fiction': 'Fiction', 'novels': 'Fiction',

    'fantasy': 'Fantasy', 'science fiction fantasy': 'Fantasy', 'magic': 'Fantasy',
    'paranormal': 'Fantasy', 'supernatural': 'Fantasy', 'urban fantasy': 'Fantasy',
    'epic fantasy': 'Fantasy', 'high fantasy': 'Fantasy', 'fairy tales': 'Fantasy',
    'mythology': 'Fantasy', 'dragons': 'Fantasy', 'witches': 'Fantasy',
    'sword and sorcery': 'Fantasy',

    'science fiction': 'Science Fiction', 'sci fi': 'Science Fiction',
    'speculative fiction': 'Science Fiction', 'dystopia': 'Science Fiction',
    'cyberpunk': 'Science Fiction', 'steampunk': 'Science Fiction',
    'space': 'Science Fiction', 'time travel': 'Science Fiction',
    'apocalyptic': 'Science Fiction', 'post apocalyptic': 'Science Fiction',
    'robots': 'Science Fiction',

    'mystery': 'Mystery & Thriller', 'thriller': 'Mystery & Thriller',
    'crime': 'Mystery & Thriller', 'suspense': 'Mystery & Thriller',
    'mystery thriller': 'Mystery & Thriller', 'detective': 'Mystery & Thriller',
    'noir': 'Mystery & Thriller', 'cozy mystery': 'Mystery & Thriller',
    'legal thriller': 'Mystery & Thriller', 'spy thriller': 'Mystery & Thriller',
    'police procedural': 'Mystery & Thriller',

    'horror': 'Horror', 'ghost stories': 'Horror', 'dark': 'Horror',
    'gothic': 'Horror', 'occult': 'Horror', 'vampires': 'Horror',
    'werewolves': 'Horror', 'zombies': 'Horror',

    'romance': 'Romance', 'contemporary romance': 'Romance',
    'paranormal romance': 'Romance', 'historical romance': 'Romance',
    'chick lit': 'Romance', 'love': 'Romance', 'erotica': 'Romance',
    'new adult': 'Romance', 'clean romance': 'Romance',

    'historical fiction': 'Historical Fiction', 'historical': 'Historical Fiction',
    'historical mystery': 'Historical Fiction', 'medieval': 'Historical Fiction',
    'civil war': 'Historical Fiction', 'world war ii': 'Historical Fiction',
    'war': 'Historical Fiction', 'ancient history': 'Historical Fiction',

    'classics': 'Classics', 'literature': 'Classics', 'british literature': 'Classics',
    'american': 'Classics', '20th century': 'Classics', '19th century': 'Classics',
    'classic literature': 'Classics',

    'young adult': 'Young Adult', 'ya': 'Young Adult', 'teen': 'Young Adult',
    'coming of age': 'Young Adult',

    'childrens': "Children's", 'middle grade': "Children's", 'juvenile': "Children's",
    'picture books': "Children's", 'kids': "Children's",

    'nonfiction': 'Nonfiction', 'non fiction': 'Nonfiction',

    'biography': 'Biography & Memoir', 'memoir': 'Biography & Memoir',
    'autobiography': 'Biography & Memoir', 'biography memoir': 'Biography & Memoir',
    'true story': 'Biography & Memoir',

    'self help': 'Self Help', 'psychology': 'Self Help', 'personal development': 'Self Help',
    'productivity': 'Self Help', 'mental health': 'Self Help', 'philosophy': 'Self Help',

    'history': 'History', 'politics': 'History',
    'military': 'History', 'military history': 'History',

    'adventure': 'Adventure', 'action': 'Adventure',
    'humor': 'Humor', 'comedy': 'Humor', 'satire': 'Humor',

    'short stories': 'Short Stories', 'anthology': 'Short Stories',
    'short story': 'Short Stories',

    'graphic novels': 'Graphic Novels', 'comics': 'Graphic Novels',
    'manga': 'Graphic Novels', 'comic book': 'Graphic Novels',
    'sequential art': 'Graphic Novels',

    'religion': 'Religion & Spirituality', 'spirituality': 'Religion & Spirituality',
    'christian': 'Religion & Spirituality', 'christian fiction': 'Religion & Spirituality',

    'science': 'Science & Nature', 'nature': 'Science & Nature',
    'environment': 'Science & Nature', 'animals': 'Science & Nature',
}

def clean_genres(genre_list):
    seen = set()
    result = []
    for g in genre_list:
        canonical = GENRE_MAP.get(g.lower().strip())
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result

df['genres_clean'] = df['genres_list'].apply(clean_genres)

# ── STEP 3: Fix descriptions ─────────────────────────────────────────────────
# Fill missing with empty string, strip junk whitespace
df['description'] = df['description'].fillna('').str.strip()
# Mark very short descriptions as empty too
df.loc[df['description'].str.len() < 20, 'description'] = ''

# ── STEP 4: Clean author (take first author only for multi-author entries) ───
# e.g. "george orwell, russell baker (preface)" → "george orwell"
df['author_clean'] = df['author'].apply(
    lambda x: x.split(',')[0].strip() if isinstance(x, str) else x
)

# ── STEP 5: Clean cover URL ───────────────────────────────────────────────────
# Some URLs have ._SX... or ._SY... size suffixes — strip them for cleaner images
import re
def clean_cover(url):
    if not isinstance(url, str): return ''
    # Remove size suffixes like ._SX318_ or ._SY475_ before .jpg
    url = re.sub(r'\._[A-Z]{2}\d+_', '', url)
    return url.strip()

df['coverImg'] = df['coverImg'].apply(clean_cover)

# ── STEP 6: Build primary genre (first genre in clean list) ──────────────────
df['primary_genre'] = df['genres_clean'].apply(lambda g: g[0] if g else 'Uncategorised')

# ── STEP 7: Drop helper columns, keep only what we need ──────────────────────
df_out = df[[
    'bookId', 'title', 'author_clean', 'rating', 'description',
    'language', 'isbn', 'genres_clean', 'primary_genre',
    'coverImg', 'title_clean', 'series'
]].rename(columns={'author_clean': 'author', 'genres_clean': 'genres'})

# ── STEP 8: Convert genres list back to JSON string for CSV storage ──────────
import json
df_out['genres'] = df_out['genres'].apply(json.dumps)

print(f"AFTER:  {len(df_out):,} books")
print(f"  Null descriptions:   {(df_out['description']=='').sum()}")
print(f"  Books without genre: {(df_out['primary_genre']=='Uncategorised').sum()}")
print(f"\nGenre distribution:")
print(df_out['primary_genre'].value_counts())

df_out.to_csv('books_final.csv', index=False)
print("\nSaved → books_final.csv")
