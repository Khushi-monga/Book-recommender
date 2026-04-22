import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

def train():
    print("Loading dataset...")
    # 1. Load data
    df = pd.read_csv('master_final.csv')
    df_rated = df[df['rating'] > 0].copy()
    
    print(f"Loaded {len(df_rated):,} explicit ratings.")
    
    # 2. To avoid extremely large sparse matrix sizes, filter for active users/books (Optional but recommended for SVD memory)
    user_counts = df_rated['user_id'].value_counts()
    book_counts = df_rated['isbn'].value_counts()
    
    # Keep users who rated >= 5 books, and books with >= 5 ratings to increase density
    valid_users = user_counts[user_counts >= 5].index
    valid_books = book_counts[book_counts >= 5].index
    
    filtered_df = df_rated[
        (df_rated['user_id'].isin(valid_users)) & 
        (df_rated['isbn'].isin(valid_books))
    ]
    print(f"Filtered to {len(filtered_df):,} ratings (Users: {filtered_df['user_id'].nunique():,}, Books: {filtered_df['isbn'].nunique():,})")
    
    # 3. Create mapping between original IDs and matrix indices
    user_ids = filtered_df['user_id'].unique()
    isbns = filtered_df['isbn'].unique()
    
    user_to_index = {user: idx for idx, user in enumerate(user_ids)}
    index_to_user = {idx: user for idx, user in enumerate(user_ids)}
    
    isbn_to_index = {isbn: idx for idx, isbn in enumerate(isbns)}
    index_to_isbn = {idx: isbn for idx, isbn in enumerate(isbns)}
    
    # 4. Create Sparse Matrix
    # We subtract average user rating from their ratings to normalize
    user_means = filtered_df.groupby('user_id')['rating'].mean()
    filtered_df['rating_norm'] = filtered_df.apply(lambda row: row['rating'] - user_means[row['user_id']], axis=1)
    
    row_indices = filtered_df['user_id'].map(user_to_index).values
    col_indices = filtered_df['isbn'].map(isbn_to_index).values
    data = filtered_df['rating_norm'].values
    
    sparse_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(len(user_ids), len(isbns)))
    
    # 5. Train SVD
    # k is the number of latent factors
    k = min(50, len(user_ids)-1, len(isbns)-1)
    print(f"Training SVD model with {k} latent factors...")
    
    # U: User feature matrix, sigma: singular values, Vt: Item feature matrix transposed
    U, sigma, Vt = svds(sparse_matrix, k=k)
    
    # Convert sigma array to diagonal matrix
    sigma_diag_matrix = np.diag(sigma)
    
    # 6. Save Model
    model_data = {
        'U': U,
        'sigma_diag': sigma_diag_matrix,
        'Vt': Vt,
        'user_to_index': user_to_index,
        'index_to_user': index_to_user,
        'isbn_to_index': isbn_to_index,
        'index_to_isbn': index_to_isbn,
        'user_means': user_means.to_dict() # Need this to restore predictions to actual 1-10 scale
    }
    
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("SVD model successfully trained and saved to svd_model.pkl!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train()
