import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv('spam.csv', encoding='latin-1')
train_data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
train_data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Create and train the model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(train_data['message'], train_data['label'])

# Create a function to predict message
def predict_message(message):
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0]
    return [probability[0] if prediction == 'ham' else probability[1], prediction]

# Test the function
print(predict_message("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."))
