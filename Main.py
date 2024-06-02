import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the data
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Clean the data
ratings = ratings[ratings.bookRating != 0]
rating_count = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(rating_count[rating_count >= 200].index)]
book_count = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(book_count[book_count >= 100].index)]

# Pivot the data
ratings_pivot = ratings.pivot(index='ISBN', columns='userID', values='bookRating').fillna(0)

# Create the model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(ratings_pivot.values)

# Function to get recommendations
def get_recommends(book = ""):
    book_info = books[books["bookTitle"] == book]
    if len(book_info) == 0:
        return f"Book '{book}' not found in the dataset."
    
    query_index = ratings_pivot.index.get_loc(book_info.iloc[0]["ISBN"])
    distances, indices = model_knn.kneighbors(ratings_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    recommended_books = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            recommended_books.append(book)
        else:
            recommended_books.append([books.iloc[indices.flatten()[i]]['bookTitle'], distances.flatten()[i]])
    
    return recommended_books

# Test the function
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
