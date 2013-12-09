import pandas as p
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from util import Util
u = Util()
from deep_net import Deep_net 
import pickle
import operator
import sys
sys.path.append('C:/cudamat/')
import pandas as p
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from util import Util
u = Util()
paths = ['C:/Data/crowdflower/train.csv', 'C:/Data/crowdflower/test.csv']
from sklearn.linear_model import MultiTaskLasso, ElasticNet, Ridge
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn import svm
from scipy import sparse
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import wordpunct_tokenize
import time
from sklearn.neural_network import BernoulliRBM
np.set_printoptions(suppress =True, precision=3)

#download wordnet
#nltk.download() 
rng = np.random.RandomState(1234)

t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

class LancasterTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.LancasterStemmer()
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]

class PorterTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.PorterStemmer()
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]

class WordnetTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc)]
    
class SnowballTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.SnowballStemmer("english")
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]  
   


tfidf1 = TfidfVectorizer(max_features=9000, strip_accents='unicode',  
    analyzer='word',token_pattern=r'\w{3,}',sublinear_tf=1,
    ngram_range=(1, 2),tokenizer=SnowballTokenizer()
    )

tfidf2 = TfidfVectorizer(max_features=40000, strip_accents='unicode',
    analyzer='char',sublinear_tf=1,
    ngram_range=(2, 17))

tfidf5 = TfidfVectorizer(max_features=9000, strip_accents='unicode',
    analyzer='char_wb',sublinear_tf=1,
    ngram_range=(2, 11))

tfidf7 = TfidfVectorizer(max_features=9000, strip_accents='unicode',  
    analyzer='word',sublinear_tf=1,#tokenizer = LancasterTokenizer(),
    ngram_range=(1, 2),tokenizer=SnowballTokenizer(),
       # stop_words = 'english',
    )

tfidf10 = TfidfVectorizer(max_features=9000, strip_accents='unicode',  
    analyzer='word',token_pattern=r'\w{1,}',sublinear_tf=1,#tokenizer = LancasterTokenizer(),
    ngram_range=(1, 2),tokenizer=LancasterTokenizer()#,stop_words = 'english'
    )



#vectorizers = [tfidf1,tfidf2,tfid3,tfid4,tfidf5,tfid6,tfidf7,tfid8,tfid9,tfidf10,tfid11,tfid12,tfid13]
#vectorizers = [tfidf1,tfid3,tfidf5,tfid6]
vectorizers = [tfidf1]
tfidf = vectorizers[0]
#comment = 'lsa = 1, tfidf2, 175000 -> 1000'
comment = 'tfidf1, with hidden multiplier'

y = np.array(t.ix[:,4:])#[:,9:]
y_original = np.array(t.ix[:,4:])#[:,9:]
cv_split = 0.2
n = int(np.round(len(t['tweet'].tolist())))
train_end = int(np.round(n*(1-cv_split)))
cv_beginning = int(np.round( n*(1-cv_split if cv_split > 0 else 0.8)))

print y.shape
print y_original.shape

train = t['tweet'].tolist()[0:train_end]
cv_X_original = np.array(t['tweet'].tolist()[cv_beginning:])
cv_y = np.array(y[cv_beginning:])
c = u.strings_to_classes(t['state'])

if cv_split == 0:
    train = t['tweet'].tolist()
else:
    y = y[0:int(np.round(len(t['tweet'].tolist())*(1-cv_split)))]   

prediction_grand_all = 0
predict_cv_grand_all = 0
list_predictions = []
list_predictions_test = []
for tfidf in vectorizers:    
    print 'fitting vectorizer...'
    tfidf.fit(t['tweet'].tolist() + t2['tweet'].tolist())
    print 'transforming train set...'
    #train = tfidf.transform(train)
    X = tfidf.transform(t['tweet'])
    
    print 'transforming cv set...'     
    cv_X = tfidf.transform(cv_X_original)
    #print 'transforming test set...'    
    test = tfidf.transform(t2['tweet']) 

vectorizers = None
tfidf = None  

print 'begin training...'
net = Deep_net([4000,4000], data = [X, y_original, test],
           set_sizes = [1-cv_split,cv_split,0],  dropout = [0.2,0.5], epochs = 75, gpu_data = True, 
           is_sparse = True, problem='regression', 
           printing = True, learning_rate = 0.1, time_printing = False, comment = comment
           #, stop_at_score = 0.13399875, ids = np.matrix(t2['id']).T, save_weights = True
           )

