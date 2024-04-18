import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle


# Read the dataset
# data = pd.read_csv("C:/Users/mvaib/OneDrive - rupee/Desktop/Ffff/fake_job_postings.csv")
data = pd.read_csv('fake_job_postings.csv')

# Select relevant columns (title, location, description, etc.) and combine them
# data['text'] = data[['title', 'location', 'company_profile', 'description', 'requirements', 'benefits']].apply(' '.join, axis=1)

# Drop unnecessary columns
# data.drop(['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits'], axis=1, inplace=True)

def split(location):
    if isinstance(location, str):
        return location.split(',')
    else:
        return [] 

print(data['location'])
data['country'] = data['location'].apply(split)
print(data['country'])

data['text'] = data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']+' '+data['industry']

del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']
del data['employment_type']


# Remove missing values (replace with blank space)
data.fillna(' ', inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['fraudulent'], test_size=0.3)

# Text vectorization (Bag-of-Words)
vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test) 

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_dtm, y_train)

# This model can now be used for predictions on new job postings
filename = 'fraudulent_job_model.pkl'
pickle.dump(model, open(filename, 'wb'))

filename = 'fraudulent_job_vectorizer.pkl'
pickle.dump(model, open(filename, 'wb'))