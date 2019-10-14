import re
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.util import ngrams
import nltk
import gensim
import json
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.decomposition import TruncatedSVD


ds = pd.read_json('News_Category_Dataset_v2.json',lines = True)
stopWords = stopwords.words('english')
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


"""********     text  cleaning        *********"""

#not useful columns
ds.drop(['link','authors','date'], axis = 1, inplace=True)



#remove emoticons from text
def remove_emoticons(text):
    t = re.sub(RE_EMOJI,"", text)
    return t

#tokenization and removing stop words, numbers and punctuation
def tokenize(text):
    tokens = [w for w in text.split() if w not in stopWords and w.isalpha()]
    if tokens == []:
          return np.nan
    return tokens

#Remove numbers
def removeNums(text):
    text = re.sub(r'\d+', '', text)
    return text

#lemmatizer
def lemmatizer(doc):
    normalized = [WordNetLemmatizer().lemmatize(word) for word in doc]
    return normalized


#stemming & postagging
porter = PorterStemmer()
def stem_postag(text):
      stems = [porter.stem(t) for t in text]
      return  stems


def clean(data):
    data = data.apply(remove_emoticons)

    #make all words in lower case
    data= data.apply(lambda x: x.lower())

    #make all words without numbers
    data  = data.apply(removeNums)

    #tokanize make all words
    data  = data.apply(tokenize)

    #remove rows that contains empty cells
    data.dropna(inplace=True)

    data = data.apply(lemmatizer)
    return data

def normalized_text(data):
    data = ' '.join(data)
    return data

#ds.short_description = clean(ds.short_description)
#ds.headline = clean(ds.headline)
#ds.dropna(inplace = True)
#ds.category = ds.category.apply(lambda x: x.lower())

ds['combined_features'] = ds['headline'] + ds['short_description']
ds.combined_features = ds.combined_features.apply(remove_emoticons)
ds.combined_features = ds.combined_features.apply(removeNums)
ds.combined_features = ds.combined_features.apply(lambda x: x.lower())
ds.combined_features = ds.combined_features.apply(tokenize)
ds.dropna(inplace=True)
ds.combined_features = ds.combined_features.apply(lemmatizer)
ds.dropna(inplace = True)
ds.category = ds.category.apply(lambda x: x.lower())

ds['normalized_combined'] = ds.combined_features.apply(normalized_text)

targets = ['travel', 'style & beauty', 'parenting']
filtered_ds = ds[ds.category.isin(targets)]


"""********     analysing empty text records       *********"""

#check weather any of records is empty
print("headlines have: {} empty records ".format(ds.headline.isnull().sum())) #754
print("short_description have: {}  empty records  ".format(ds.short_description.isnull().sum())) #23877

empty_headlines = ds[ds.headline.isnull()]

print("short_description and headline have: {}  empty records  "
      .format(empty_headlines.short_description.isnull().sum())) #106


#counts of labels in dataset
print("counts of labels in dataset:")
print(ds.category.value_counts())

#since the worldpost == worldpost
ds.category.replace('the worldpost','worldpost', inplace = True)

#count number of empty values from each label
classes = ds.category.unique()
empty_cells = pd.DataFrame(columns = ['label', 'headline', 'shortDesc', 'both'])
counter = 0
for label in classes:
    n1 = ds[ds['category'] == label].headline.isnull().sum()
    n2 = ds[ds['category'] == label].short_description.isnull().sum()
    n3 = empty_headlines[empty_headlines['category'] == label].short_description.isnull().sum()
    empty_cells.loc[counter] = [label, n1,n2,n3]
    counter = counter + 1


"""********    feature extraction    *********"""

#print('Total time: ' + str((time.time() - start)) + ' secs')

# Create CBOW model
model_CBOW1 = gensim.models.Word2Vec(filtered_ds['combined_features'],
                                     min_count = 50, size = 50, window = 3)
model_CBOW2 = gensim.models.Word2Vec(filtered_ds['combined_features'],
                                     min_count = 100, size = 70, window = 5)
model_CBOW3 = gensim.models.Word2Vec(filtered_ds['combined_features'],
                                     min_count = 250, size = 50, window = 7)


print("CBOW1 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_CBOW1.wv.similarity('health', 'care'))

print("CBOW2 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_CBOW2.wv.similarity('health', 'care'))

print("CBOW3 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_CBOW3.wv.similarity('health', 'care'))
# Create SkipGram model
model_SkipGram1 = gensim.models.Word2Vec(filtered_ds["combined_features"],
                                         min_count = 50, size = 100,
                                         window = 5, sg = 1)
model_SkipGram2 = gensim.models.Word2Vec(filtered_ds["combined_features"],
                                         min_count = 100, size = 70,
                                         window = 5, sg = 1)
model_SkipGram3 = gensim.models.Word2Vec(filtered_ds["combined_features"],
                                         min_count = 250, size = 100,
                                         window = 7, sg = 1)

# Print results
print("SkipGram1 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_SkipGram1.wv.similarity('health', 'care'))

print("SkipGram3 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_SkipGram2.wv.similarity('health', 'care'))

print("SkipGram3 Cosine similarity between 'health' " +
               "and 'care' : ",
   model_SkipGram3.wv.similarity('health', 'care'))

#create a dictionary mapping word to build features
w2v_CBOW= dict(zip(model_CBOW3.wv.index2word, model_CBOW3.wv.vectors))


"""********    KNN Classification   *********"""

#split the dataset 30-70
x = filtered_ds['normalized_combined']
y = filtered_ds['category']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

#Create KNN Classifier using pipeline to combine it with w2v
knn = Pipeline([
    ("word2vec vectorizer", TfidfVectorizer(w2v_CBOW)),
    ("KNeighborsClassifiers",  KNeighborsClassifier(weights  = 'distance'))])

#define tunning parameters
k = np.array([7, 9, 11, 13, 15])

parameters = {'KNeighborsClassifiers__n_neighbors': k}

#apply grid search to find best parameters using the training set
clf = GridSearchCV(knn, parameters, cv=5, scoring='accuracy', verbose=5, n_jobs=3, return_train_score = True)
clf.fit(X_train,y_train)
#print the best results
print("GridSearch best parameter k: {} ".format(clf.best_params_['KNeighborsClassifiers__n_neighbors']))
print("GridSearch best training accuracy: {}% ".format(clf.best_score_*100))

#plot all the results
plt.plot(k, clf.cv_results_.get('mean_test_score'))
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#validate the model using the testing set
optimised_knn = clf.best_estimator_

oknn_test = optimised_knn.score(X_test, y_test)
print("KNN validation accuracy: {} ".format(oknn_test*100))

#evaluation on test data
pred = optimised_knn.predict(X_test)
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
confusion_matrix(pred, y_test)
recall_score(y_true=y_test, y_pred=pred, average='macro', labels = unique_elements)
#save the best accuracy model
filename = '15_final_knn_model.pkl'
joblib.dump(clf.best_estimator_, filename, compress = 1)

#load the saved model
knn = joblib.load('15_final_knn_model.pkl')

"""********    SVM Classification   *********"""
#resources

#create a SVM classfier using a piplinw
Svm= Pipeline([("word2vec vectorizer", TfidfVectorizer(w2v_CBOW)), ('SVM', SVC())])

#define the parameters
c= np.array([0.001,0.1,10,15,100])
gamma=np.array([0.1,0.01])


parameteres = {'SVM__C':c, 'SVM__gamma':gamma}
#applying gridsearch
grid = GridSearchCV(Svm, param_grid=parameteres, cv=3,scoring='accuracy', verbose=3)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print( "score = %3.2f" %(grid.score(X_test,y_test)))
print( "best score = %3.2f" %(grid.best_score_*100))
print(grid.best_params_)
print("Recall score: {} ".format(recall_score(y_test, y_pred , average = 'macro' )))
print("Confusion Matrix:")
matrix2 = confusion_matrix(y_test, y_pred,labels= targets)
print('travel , style & beauty , parenting')
for i in range(3):
    print(targets[i]+" : {0},{1},{2}".format(matrix2[i][0],matrix2[i][1],matrix2[i][2]))


#clf = svm.SVC(kernel='linear')
#clf.fit(x_train, y_train)
#y_pred = clf.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))


"""********    multinomial Naive bayes Classification   *********"""

MultiNB = MultinomialNB()

MultiNB = Pipeline([('vect', TfidfVectorizer()), 

    ('clf', MultinomialNB()) ])

MultiNB.fit(X_train,y_train)

print("model record and it's parameters: ")
print(MultiNB)
predicted = MultiNB.predict(X_test)
print('Accuracy achieved is ' + str(np.mean(predicted == y_test)))
print('Precision achieved is :'+ str(precision_score(y_test, predicted , average='macro')))
print("Recall:"+str(recall_score(y_test, predicted , average='macro')))




"""********    Gradient Boosting Classification   *********"""
#Create Bagging Classifier using pipeline to combine it with w2v 
pipe = Pipeline([("word2vecVectorizer", TfidfVectorizer(w2v_CBOW)),
 ("boostingClassifier",  GradientBoostingClassifier())])



#0.25,0.1,0.01
lr = [0.25,0.1]
##50..600
n_model = [400,600,700]
##1..10
max_depth = [4,6,10]
##1..8
min_leaf =[4,6,8]
param_grid = {'boostingClassifier__criterion':['friedman_mse'],
              'boostingClassifier__init': [None],
              'boostingClassifier__learning_rate':[0.25],
              'boostingClassifier__max_depth':[6],
              'boostingClassifier__n_estimators':[600],
              'boostingClassifier__min_samples_leaf':[6],
              'boostingClassifier__min_samples_split':[0.1]}   

clf = GridSearchCV(pipe,param_grid, cv=5, scoring='accuracy', verbose=5, n_jobs = -1) 
clf.fit(X_train,y_train)


print()
print()
print()
print("-------------- Results -------------")

###print the best results
print("GridSearch best parameter: {} ".format(clf.best_params_)) 
print("GridSearch best accuracy: {}% ".format(clf.best_score_*100)) 

#validate the model using the testing set

print("GridSearch validation accuracy: {} ".format(clf.score(X_test, y_test))) 
y_pred = clf.predict(X_test)
print("recall score: {} ".format(recall_score(y_test, y_pred , average = 'macro'))) 
print("Confusion Matrix: ")


matrix2 = confusion_matrix(y_test, y_pred,labels= targets)
print('travel , style & beauty , parenting')
for i in range(3):
    print(targets[i]+" : {0},{1},{2}".format(matrix2[i][0],matrix2[i][1],matrix2[i][2]))
   

