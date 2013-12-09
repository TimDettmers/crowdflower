import pickle
import pandas as p
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from util import Util
u = Util()
from sklearn.linear_model import MultiTaskLasso, ElasticNet, Ridge
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn import svm
from scipy import sparse
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

import time
import nltk
from nltk.tokenize import wordpunct_tokenize
np.set_printoptions(suppress =True, precision=3)
#import logging as log #install the logging module to enable logging of the results
#log.basicConfig(filename='C:/results.txt', format='%(message)s', level=log.DEBUG)



def main():
    paths = ['C:/Data/crowdflower/train.csv', 'C:/Data/crowdflower/test.csv']    
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
    
    
    tfidf1 = TfidfVectorizer(max_features=85000, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{3,}',sublinear_tf=1,
        ngram_range=(1, 2),tokenizer = SnowballTokenizer())
    
    tfidf2 = TfidfVectorizer(max_features=600000, strip_accents='unicode',
        analyzer='char',sublinear_tf=1,
        ngram_range=(2, 17))
    
    tfidf3 = CountVectorizer(max_features=5200, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{3,}',
        ngram_range=(1, 3),tokenizer = SnowballTokenizer())
    
    tfidf4 = CountVectorizer(max_features=1800, strip_accents='unicode',  
        analyzer='char',
        ngram_range=(2, 9))
    
    tfidf5 = TfidfVectorizer(max_features=10000, strip_accents='unicode',
        analyzer='char_wb',sublinear_tf=1,
        ngram_range=(2, 9))
    
    tfidf6 = CountVectorizer(max_features=1800, strip_accents='unicode',  
        analyzer='char_wb',
        ngram_range=(2, 9))
    
    tfidf7 = TfidfVectorizer(max_features=85000, strip_accents='unicode',  
        analyzer='word',sublinear_tf=1,
        ngram_range=(1, 2),tokenizer = SnowballTokenizer())
    
    tfidf8 = CountVectorizer(max_features=4900, strip_accents='unicode',  
        analyzer='word',
        ngram_range=(1, 3),tokenizer = SnowballTokenizer())
    
    tfidf9 = CountVectorizer(max_features=5200, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 3),tokenizer = SnowballTokenizer())
    
    tfidf10 = TfidfVectorizer(max_features=85000, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',sublinear_tf=1,
        ngram_range=(1, 2),tokenizer = SnowballTokenizer())
    
    tfidf11 = CountVectorizer(max_features=5200, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{3,}', binary=True,
        ngram_range=(1, 3),tokenizer = SnowballTokenizer())
    
    tfidf12 = CountVectorizer(max_features=5200, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}', binary=True,
        ngram_range=(1, 3),tokenizer = SnowballTokenizer()
        )
    
    tfidf13 = CountVectorizer(max_features=5200, strip_accents='unicode',  
        analyzer='word', binary=True,
        ngram_range=(1, 3),tokenizer = SnowballTokenizer())
    
    vectorizers = [tfidf1,tfidf2,tfidf3,tfidf4,tfidf5,tfidf6, tfidf7,tfidf8,tfidf9,tfidf10,tfidf11,tfidf12,tfidf13]
    #vectorizers = [tfidf1,tfidf3,tfidf5,tfidf6]
    #vectorizers = [tfidf1]  
      
    #comment = 'full, SnowballTokenizer no RF'
    use_lsa = 0
    cv_split = 0
    n = int(np.round(len(t['tweet'].tolist())))
    train_end = int(np.round(n*(1-cv_split)))
    cv_beginning = int(np.round( n*(1-cv_split
                                     if cv_split > 0 else 0.8)))
    y = np.array(t.ix[:,4:])
    
    train = t['tweet'].tolist()[0:train_end]
    cv_X_original = np.array(t['tweet'].tolist()[cv_beginning:])
    cv_y = np.array(y[cv_beginning:])
        
    if cv_split == 0:
        train = t['tweet'].tolist()
    else:
        y = y[0:int(np.round(len(t['tweet'].tolist())*(1-cv_split)))]   
    
    prediction_grand_all = 0
    predict_cv_grand_all = 0
    list_predictions = []
    list_predictions_test = []
    for tfid in vectorizers:    
        print 'fitting vectorizer...'
        tfid.fit(t['tweet'].tolist() + t2['tweet'].tolist())
        print 'transforming train set...'
        X = tfid.transform(train)
        print 'transforming cv set...'     
        cv_X = tfid.transform(cv_X_original)
        print 'transforming test set...'    
        test = tfid.transform(t2['tweet'])    
        
        clf1 = MultiTaskLasso()
        clf2 = AdaBoostRegressor(learning_rate = 1,n_estimators = 10)
        clf3 = RandomForestRegressor(max_depth = 20, n_estimators = 36, max_features = 100, n_jobs = 6)
        clf4 = Ridge()       
       
        clfs = [clf4]
        lsa_classifier = [0]
        prediction_all = 0
        predict_cv_all = 0
        for clf, use_lsa in zip(clfs,lsa_classifier):            
            if use_lsa == 1:
                lsa = TruncatedSVD(n_components = 100)
                print 'fitting lsa...'
                lsa.fit(X, y)
                print 'transfomring with lsa...'
                X = lsa.transform(X)
                cv_X = lsa.transform(cv_X)
                test = lsa.transform(test)                
                print 'normalizing....'
                norm = Normalizer()
                norm.fit(X, y)
                X = norm.transform(X, copy= False)
                test = norm.transform(test, copy= False)
                cv_X = norm.transform(cv_X, copy= False)   
                
            else:           
                fac = p.Categorical(t['state'].tolist() + t2['state'].tolist())        
                t_matrix = u.create_t_matrix(fac.labels)
                train_feat = t_matrix[0:train_end]
                cv_X_feat = t_matrix[cv_beginning:n]
                test_feat = t_matrix[n:]                        
                X = sparse.hstack([X, sparse.csr_matrix(train_feat)])
                cv_X = sparse.hstack([cv_X, sparse.csr_matrix(cv_X_feat)])
                test = sparse.hstack([test, sparse.csr_matrix(test_feat)])     
            
          
            t0 = time.time()
            print 'fitting...'
            clf.fit(X,y) 
            print 'validating...'
            print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/ (X.shape[0]*24.0)))
            
            prediction = np.array(clf.predict(test))  
            prediction = np.abs(prediction*(prediction > 0))
            prediction[prediction > 1] = 1
            
            predict_cv = np.array(clf.predict(cv_X))  
                    
            predict_cv = np.abs(predict_cv*(predict_cv > 0))
            predict_cv[predict_cv > 1] = 1
            list_predictions.append(predict_cv)
            list_predictions_test.append(prediction)
            print 'Cross validation error: {0}'.format(np.sqrt(np.sum(np.array(predict_cv-cv_y)**2)/ (cv_X.shape[0]*24.0)))
            predict_cv_all = predict_cv + predict_cv_all
            prediction_all = prediction + prediction_all
            print 'fitted model in {0} seconds'.format(np.round(time.time() - t0,2))
        prediction_all /= len(clfs)*1.0
        predict_cv_all /= len(clfs)*1.0
        print 'Cross validation error ensemble: {0}'.format(np.sqrt(np.sum(np.array(predict_cv_all - cv_y)**2)/ (cv_X.shape[0]*24.0)))
        prediction_grand_all = prediction_all + prediction_grand_all
        predict_cv_grand_all = predict_cv_all + predict_cv_grand_all
    
    
    prediction_grand_all /= len(vectorizers)*1.0
    predict_cv_grand_all /= len(vectorizers)*1.0
    
    print 'Cross validation error grand ensemble: {0}'.format(np.sqrt(np.sum(np.array(predict_cv_grand_all - cv_y)**2)/ (cv_X.shape[0]*24.0)))
    #log.info(comment)
    #log.info(np.sqrt(np.sum(np.array(predict_cv_grand_all - cv_y)**2)/ (cv_X.shape[0]*24.0)))
    prediction = np.array(np.hstack([np.matrix(t2['id']).T, prediction_grand_all])) 
    col = '%i,' + '%f,'*23 + '%f'
    np.savetxt('C:/data/crowdflower/sklearn_prediction.csv', prediction,col, delimiter=',')  
    list_predictions.append(cv_y)
    pickle.dump(list_predictions, io.open('C:/data/crowdflower/predicts.txt','wb'))
    pickle.dump(list_predictions_test, io.open('C:/data/crowdflower/predicts_test.txt','wb'))

if __name__=="__main__":
    main()