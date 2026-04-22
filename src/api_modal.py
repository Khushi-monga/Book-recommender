from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import pickle

app = Flask(__name__)
CORS(app)

print("Loading dataset...")

# Load full dataset — keep rating>0 rows for collaborative data
df = pd.read_csv('master_final.csv')
df_rated = df[df['rating'] > 0].copy()

df_rated['author_clean'] = df_rated['author'].apply(
    lambda x: re.sub(r'\s*\(goodreads author\)', '', str(x)).strip().title()
)

# ── BOOK INFO: load from FULL df so HP catalogue entries (rating=0) are included ──
df_all = df.copy()
df_all['author_clean'] = df_all['author'].apply(
    lambda x: re.sub(r'\s*\(goodreads author\)', '', str(x)).strip().title()
)

book_info = df_all.drop_duplicates(subset='isbn')[[
    'isbn', 'title', 'author', 'author_clean', 'primary_genre',
    'coverImg', 'book_avg_rating', 'description'
]].reset_index(drop=True)

# ── RATINGS AGG: only from real ratings (rating > 0) ──
ratings_agg = df_rated.groupby('isbn').agg(
    avg_rating  = ('rating', 'mean'),
    num_ratings = ('rating', 'count')
).reset_index()
ratings_agg['avg_rating'] = ratings_agg['avg_rating'].round(2)

# ── POPULARITY: needs real ratings ──
popularity = ratings_agg.merge(book_info, on='isbn', how='left')
popularity = popularity[
    (popularity['num_ratings'] >= 5) &
    (popularity['avg_rating']  >= 7)
].sort_values('num_ratings', ascending=False).dropna(subset=['title'])

print(f"Rated rows: {len(df_rated):,} | Total books: {book_info.shape[0]:,} | "
      f"Users: {df_rated['user_id'].nunique():,} | Trending pool: {len(popularity):,}")

# ── LOAD SVD MODEL ──
svd_model = None
try:
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    print("SVD Model loaded successfully!")
except Exception as e:
    print(f"Could not load SVD model (You must run train_svd.py to generate it): {e}")

print("Ready ✓")


def clean_record(row, score=None):
    desc  = str(row.get('description', '') or '')
    cover = str(row.get('coverImg', '') or '')
    # Use user-ratings avg first, fall back to book_avg_rating from Goodreads
    avg = float(row.get('avg_rating') or row.get('book_avg_rating') or 0)
    # Short description for card (~150 chars, break at word boundary)
    short = (desc[:150].rsplit(' ', 1)[0] + '...') if len(desc) > 150 else desc
    
    # Optional match score for collaborative filtering
    match_score = score if score is not None else 0
    
    return {
        'title':         str(row.get('title', '')).title(),
        'author':        str(row.get('author_clean', row.get('author', ''))).title(),
        'primary_genre': str(row.get('primary_genre', '')),
        'avg_rating':    round(avg, 2),
        'num_ratings':   int(row.get('num_ratings', 0) or 0),
        'coverImg':      cover,
        'description':   short,
        'score':         match_score
    }


@app.route('/api/trending')
def trending():
    n = request.args.get('n', 20, type=int)
    return jsonify([clean_record(r) for _, r in popularity.head(n).iterrows()])


@app.route('/api/search')
def search():
    q = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'all').lower()
    n = request.args.get('n', 20, type=int)
    
    if not q:
        return jsonify([])
        
    mask = pd.Series([False] * len(book_info))
    
    # Generic content-based text matching across relevant columms
    if search_type in ['title', 'all']:
        mask = mask | book_info['title'].str.contains(q, case=False, na=False, regex=False)
    if search_type in ['author', 'all']:
        mask = mask | book_info['author_clean'].str.contains(q, case=False, na=False, regex=False)
    if search_type in ['genre', 'all']:
        mask = mask | book_info['primary_genre'].str.contains(q, case=False, na=False, regex=False)
        
    matched = book_info[mask].copy()
    if matched.empty:
        return jsonify([])

    result = matched.merge(ratings_agg, on='isbn', how='left')
    result['avg_rating']  = result['avg_rating'].fillna(result['book_avg_rating'])
    result['num_ratings'] = result['num_ratings'].fillna(0)

    result = result.sort_values('num_ratings', ascending=False).head(n)
    return jsonify([clean_record(r) for _, r in result.iterrows()])


@app.route('/api/by-author')
def by_author():
    author = request.args.get('author', '').strip()
    n      = request.args.get('n', 20, type=int)
    if not author:
        return jsonify([])

    # Match against book_info
    mask = book_info['author_clean'].str.contains(author, case=False, na=False, regex=False)
    matched = book_info[mask].copy()

    if matched.empty:
        return jsonify([])

    result = matched.merge(ratings_agg, on='isbn', how='left')
    result['avg_rating']  = result['avg_rating'].fillna(result['book_avg_rating'])
    result['num_ratings'] = result['num_ratings'].fillna(0)

    result = result.sort_values('avg_rating', ascending=False).head(n)
    return jsonify([clean_record(r) for _, r in result.iterrows()])


@app.route('/api/by-genre')
def by_genre():
    genre = request.args.get('genre', '').strip()
    n     = request.args.get('n', 20, type=int)
    if not genre:
        return jsonify([])

    GENRE_MAP = {
        'Fantasy':      'Fantasy',
        'Sci-Fi':       'Science Fiction',
        'Mystery':      'Mystery & Thriller',
        'Thriller':     'Mystery & Thriller',
        'Classic':      'Classics',
        'Romance':      'Romance',
        'Non-fiction':  'Nonfiction',
        'Adventure':    'Adventure',
        'Horror':       'Horror',
        'Biography':    'Biography & Memoir',
        'History':      'Historical Fiction',
        'Young Adult':  'Young Adult',
        'Science Fiction': 'Science Fiction',
        'Historical Fiction': 'Historical Fiction',
        'Nonfiction':   'Nonfiction',
    }
    mapped = GENRE_MAP.get(genre, genre)
    mask   = book_info['primary_genre'].str.contains(mapped, case=False, na=False)
    result = book_info[mask].merge(ratings_agg, on='isbn', how='left')
    result['avg_rating']  = result['avg_rating'].fillna(result['book_avg_rating'])
    result['num_ratings'] = result['num_ratings'].fillna(0)
    result = result[result['num_ratings'] >= 3].sort_values('avg_rating', ascending=False).head(n)
    return jsonify([clean_record(r) for _, r in result.iterrows()])


@app.route('/api/authors')
def authors():
    q = request.args.get('q', '').strip().lower()
    all_authors = sorted(set(book_info['author_clean'].dropna().tolist()))
    if q:
        all_authors = [a for a in all_authors if q in a.lower()]
    return jsonify(all_authors[:20])


@app.route('/api/for-you')
def for_you():
    if not svd_model:
        return jsonify({"error": "Model not loaded"}), 500
        
    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', 20, type=int)
    
    if not user_id or user_id not in svd_model['user_to_index']:
        # If unknown user, fallback to trending
        return jsonify([clean_record(r) for _, r in popularity.head(n).iterrows()])

    # SVD User Personalization
    u_idx = svd_model['user_to_index'][user_id]
    user_vec = svd_model['U'][u_idx, :] @ svd_model['sigma_diag']
    
    # Predict all book ratings
    preds = user_vec @ svd_model['Vt']
    preds += svd_model['user_means'][user_id] # Restore base mean rating level
    
    # We shouldn't recommend books they've already rated
    rated_isbns = df_rated[df_rated['user_id'] == user_id]['isbn'].unique()
    rated_indices = [svd_model['isbn_to_index'][isbn] for isbn in rated_isbns if isbn in svd_model['isbn_to_index']]
    
    if rated_indices:
        preds[rated_indices] = -np.inf
        
    # Get top predicted books
    top_k_indices = preds.argsort()[::-1][:n]
    top_isbns = [svd_model['index_to_isbn'][idx] for idx in top_k_indices]
    
    # Normalizing match score for UI presentation (percentage base)
    max_score, min_score = np.max(preds[top_k_indices]), np.min(preds[preds > -np.inf])
    
    # Retrieve book specs and merge rating aggregates
    rec_df = book_info[book_info['isbn'].isin(top_isbns)].copy()
    rec_df = rec_df.merge(ratings_agg, on='isbn', how='left')
    rec_df['avg_rating']  = rec_df['avg_rating'].fillna(rec_df['book_avg_rating'])
    rec_df['num_ratings'] = rec_df['num_ratings'].fillna(0)
    
    def generate_record(row):
        # find original rank to match index alignment
        idx = svd_model['isbn_to_index'][row['isbn']]
        raw_pred = preds[idx]
        
        # Pseudo-normalization out of 100% just for UI flavor 
        if max_score > min_score:
             norm_score = max(0.60, min(0.99, (raw_pred - min_score) / (max_score - min_score)))
        else:
             norm_score = 0.95
             
        return clean_record(row, score=norm_score)

    # Sort matching predictions in descending order of SVD top ranks
    rec_df['svd_rank'] = rec_df['isbn'].apply(lambda x: top_isbns.index(x) if x in top_isbns else 999)
    rec_df = rec_df.sort_values('svd_rank')

    return jsonify([generate_record(r) for _, r in rec_df.iterrows()])


if __name__ == '__main__':
    app.run(debug=True, port=5000)


@app.route('/api/book-detail')
def book_detail():
    """Returns full book info including complete description — used by modal."""
    title = request.args.get('title', '').strip()
    isbn  = request.args.get('isbn', '').strip()
    if not title and not isbn:
        return jsonify({}), 400

    if isbn:
        mask = book_info['isbn'].astype(str) == str(isbn)
    else:
        mask = book_info['title'].str.lower() == title.lower()

    match = book_info[mask]
    if match.empty:
        # fuzzy fallback — contains
        mask2 = book_info['title'].str.contains(title, case=False, na=False, regex=False)
        match = book_info[mask2].head(1)

    if match.empty:
        return jsonify({}), 404

    row   = match.iloc[0]
    rdata = ratings_agg[ratings_agg['isbn'] == row['isbn']]
    avg   = rdata['avg_rating'].values[0] if not rdata.empty else row['book_avg_rating']
    cnt   = int(rdata['num_ratings'].values[0]) if not rdata.empty else 0

    desc = str(row.get('description', '') or '')
    return jsonify({
        'title':         str(row['title']).title(),
        'author':        str(row.get('author_clean', row['author'])).title(),
        'primary_genre': str(row.get('primary_genre', '')),
        'avg_rating':    round(float(avg or 0), 2),
        'num_ratings':   cnt,
        'coverImg':      str(row.get('coverImg', '') or ''),
        'description':   desc,   # FULL description, no truncation
    })
