from textblob import TextBlob
import pandas as pd
import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import spatial
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from pandarallel import pandarallel
import re
from nltk.stem import WordNetLemmatizer 
from termcolor import colored
from scipy.stats import spearmanr

My_Folder_Path='/content/drive/MyDrive/Language Bias/'
My_Folder_Path=''

import pickle

with open('Datasets/Hate_Data/hate_detector_model', 'rb') as f:
    vect, tf_transformer, clf = pickle.load(f)

def hash(astring):
  return ord(astring[0])


lemmatizer = WordNetLemmatizer()
punctuations = """!"$%()*+,*/:#»«'";“<=>?[\]^`”{|}~"""
def clean_text(row):
    row=re.sub('\u200c','',row)
    row=re.sub('\n\n',' . ',row)
    row=re.sub('\n',' ',row)
    row=re.sub("-?NEWLINE_TOKEN", " ",row)
    row=re.sub("TAB_TOKEN", " ",row)
    row=re.sub("Alternate option=", "",row)
    row=re.sub('RT','',row)
    row = row.lower()
    row=re.sub("@[A-Za-z0-9]+","",row)
    row=re.sub("http\S+|www.\S+","",row)
    row = re.sub(r'<.*?>', '', row)
    row=re.sub("&amp","&",row)
    row=re.sub("\d","",row)
    row = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(row)])
    no_punct = ""
    for char in row:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def TrainModel(df, df_comment_column='body', outputname='outputModel', window = 5, minf=10, epochs=10, ndim=200, lemmatiseFirst = False, verbose = True):
  def loadCSVAndPreprocess(df, column = 'body', nrowss=None, verbose = True):
    '''
		input:
		path <str> : path to csv file
		column <str> : column with text
		nrowss <int> : number of rows to process, leave None if all
		verbose <True/False> : verbose output
		tolower <True/False> : transform all text to lowercase
		returns:
		list of preprocessed sentences
		'''
    def preprocessor(row):
      row = gensim.utils.simple_preprocess(row)
      return row
    
    documents=df[df_comment_column].apply(lambda x: preprocessor(clean_text(x)))
    print('Done reading all documents')
    return documents
  
  def trainWEModel(documents, outputfile, ndim, window, minfreq, epochss):
    '''
		documents list<str> : List of texts preprocessed
		outputfile <str> : final file will be saved in this path
		ndim <int> : embedding dimensions
		window <int> : window when training the model
		minfreq <int> : minimum frequency, words with less freq will be discarded
		epochss <int> : training epochs
		'''
    starttime = time.time()
    print('->->Starting training model {} with dimensions:{}, minf:{}, epochs:{}'.format(outputfile,ndim, minfreq, epochss))
    model = gensim.models.Word2Vec (documents, vector_size=ndim, window=window, min_count=minfreq, workers=1,seed=1,hashfxn=hash, sg=0)
    model.train(documents,total_examples=len(documents),epochs=epochss)
    model.save(outputfile)
    print('->-> Model saved in {}'.format(outputfile))

     
  print('->Starting with Data-Frame [{}], output {}, window {}, minf {}, epochs {}, ndim {}'.format(df_comment_column,outputname, 
                                                                                            window, minf, epochs, ndim))
  docs = loadCSVAndPreprocess(df, df_comment_column, nrowss=None, verbose=verbose)
  starttime = time.time()
  print('-> Output will be saved in {}'.format(outputname))
  trainWEModel(docs, outputname, ndim, window, minf, epochs)
  print('-> Model creation ended in {} seconds'.format(time.time()-starttime))


sid = SentimentIntensityAnalyzer()
def GetTopMostBiasedWords(model,adjectives_list_path, c1, c2, pos = ['JJ','JJR','JJS'], verbose = True):
  '''
	modelpath <str> : path to skipgram w2v model
	topk <int> : topk words
	c1 list<str> : list of words for target set 1
	c2 list<str> : list of words for target set 2
	pos list<str> : List of parts of speech we are interested in analysing
	verbose <bool> : True/False
  '''
  def calculateCentroid(model, words, words_sorted):
    embeddings = [np.array(model.wv.__getitem__(w)) for w in words if model.wv.__contains__(w)]
    words_sorted_df=pd.DataFrame(words_sorted)
    weights = [words_sorted_df.loc[words_sorted_df[0]==w,2] for w in words if model.wv.__contains__(w)]
    weights=np.asarray(weights).reshape(len(weights))
    centroid = np.zeros(len(embeddings[0]))
    sum_weights=0
    for i,e in enumerate(embeddings):
        centroid += e*weights[i]
        sum_weights += weights[i]       
    return centroid/sum_weights


  def getCosineDistance(embedding1, embedding2):
    return spatial.distance.cosine(embedding1, embedding2)


	#select the interesting subset of words based on pos
  words_sorted = sorted( [(k,v, model.wv.get_vecattr(k, "count")) for (k,v) in model.wv.key_to_index.items()] ,  key=lambda x: x[1], reverse=False)
  words = [w for w in words_sorted if nltk.pos_tag([w[0]])[0][1] in pos]

  if len(c1) < 1 or len(c2) < 1 or len(words) < 1:
    print('[!] Not enough word concepts to perform the experiment')
    return None

  centroid1, centroid2 = calculateCentroid(model, c1,words_sorted),calculateCentroid(model, c2,words_sorted)
  winfo = []
  for i, w in enumerate(words):
    word = w[0]
    freq = w[2]
    rank = w[1]
    pos = nltk.pos_tag([word])[0][1]
    wv = model.wv.__getitem__(word)
    sentNLTK = sid.polarity_scores(word)['compound']
    sentTextBlob = TextBlob(word).sentiment.polarity
    #estimate cosinedistance diff
    d1 = getCosineDistance(centroid1, wv)
    d2 = getCosineDistance(centroid2, wv)
    bias = d2-d1

    winfo.append({'word':word, 'bias':bias, 'freq':freq,
                  'pos':pos, 'wv':wv, 'rank':rank,
                  'sentNLTK':sentNLTK, 'sentTextBlob':sentTextBlob} )

    if(i%100 == 0 and verbose == True):
      print('...'+str(i), end="")

	#Get max and min topk biased words...
  biasc1 = sorted( winfo, key=lambda x:x['bias'], reverse=True )#[:min(len(winfo), topk)]
  biasc2 = sorted( winfo, key=lambda x:x['bias'], reverse=False )#[:min(len(winfo), topk)]
    #move the ts2 bias to the positive space
  for w2 in biasc2:
    w2['bias'] = w2['bias']*-1
  
  pd.DataFrame(biasc2).to_excel(adjectives_list_path+'.xlsx', index=None)

  return [biasc1, biasc2]




def polarity_extractor (adj_list,dict_key,dict_value,corpus_df, text_column,dict_save_path):
  
#   corpus_df=pd.read_csv(corpus_df_path,usecols=[text_column],lineterminator='\n').fillna('')
#   print('successfully read the corpus from {}'.format(corpus_df_path))
#   corpus_df = corpus_df[corpus_df[text_column].map(len)>10].sample(n=n_samples, random_state=1,replace=True)
#   print('Removed Null Values And Sampled {} Rows'.format(n_samples))

  pandarallel.initialize(nb_workers=60)
  corpus_df[text_column]=corpus_df[text_column].parallel_apply(lambda x: clean_text(x))
  print('successfully cleansed and lemmatized the corpus')
  print('successfully read the dictionary')
  separators = "."
  def custom_split(sepr_list, str_to_split):
    regular_exp = '|'.join(map(re.escape, sepr_list))
    return re.split(regular_exp, str_to_split)

  
  corpus_df=pd.DataFrame(custom_split(separators, corpus_df[text_column].str.cat(sep='.')))
  corpus_df.columns=[text_column]
  print('Done tokenizing sentences')

  
  sid = SentimentIntensityAnalyzer()
  corpus_df['sentence polarity']=corpus_df[text_column].parallel_apply(lambda x: (TextBlob(x).sentiment.polarity+sid.polarity_scores(x)['compound'])/2)
  corpus_df['sentence hate']=corpus_df[text_column].parallel_apply(lambda x: clf.predict(tf_transformer.transform(vect.transform([x])))[0])
  

  print('Assigned a polarity to each sentence')
  
  
  
  def adjective_polarity(corpus_df_X,text_column_X,word_X):
    # contains_word_X = corpus_df_X[corpus_df_X[text_column_X].str.contains(r'\b{}\b'.format(word_X))]['sentence polarity']
    contains_word_X = corpus_df_X[corpus_df_X[text_column_X].map(lambda x: word_X in x.split())][['sentence polarity','sentence hate']]
    return pd.Series([contains_word_X['sentence polarity'].mean(),contains_word_X['sentence hate'].mean(), contains_word_X['sentence hate'].count()])
  
  adj_list[[dict_value,'hate' ,dict_value+'_frequency']]=adj_list[dict_key].parallel_apply(lambda x: adjective_polarity(corpus_df_X=corpus_df, text_column_X = text_column, word_X = x))#, axis=1)
  print('Assigned a polarity to each word')

  # adj_list[dict_value]=adj_list[dict_key].swifter.apply(lambda x: corpus_df[corpus_df[text_column].str.contains(x)]['sentence polarity'].mean())
  # adj_list[dict_value+' frequency']=adj_list[dict_key].swifter.apply(lambda x: corpus_df[corpus_df[text_column].str.contains(x)]['sentence polarity'].count())
  # print('Assigned a polarity to each word')
  # adj_list[dict_value + ' TextBlob default']=adj_list[dict_key].swifter.apply(lambda x: TextBlob(x).sentiment.polarity)
  # sid = SentimentIntensityAnalyzer()
  # adj_list[dict_value + ' NLTK default']=adj_list[dict_key].swifter.apply(lambda x: sid.polarity_scores(x)['compound'])
  # print('Added TextBlob and NLTK default polarities')




  adj_list.to_excel(dict_save_path,index=False)
  print('The list of adjectives with their polarities saved in {}'.format(dict_save_path))

  return adj_list

        
def polarity_normalizer(my_adj_polarity_path,my_col, new_col):
    df=pd.read_excel(my_adj_polarity_path)
#     df=df[df['freq']>10]
    df=df.sort_values(by='sentiment').reset_index(drop=True)
    zero_index=round(len(df[(df['sentTextBlob']+df['sentNLTK'])<0])/(len(df[(df['sentTextBlob']+df['sentNLTK'])<0])+len(df[(df['sentTextBlob']+df['sentNLTK'])>0]))*len(df))-1
    df['sentiment_normalized']=df['sentiment']-df['sentiment'][zero_index]
    
    
    
    df_neg=df[0:zero_index]
    df_pos=df[zero_index:]

    upper_bound_neg=np.percentile(df_neg[my_col].dropna(),75)+1.5*(np.percentile(df_neg[my_col].dropna(),75)-np.percentile(df_neg[my_col].dropna(),25))
    lower_bound_neg=np.percentile(df_neg[my_col].dropna(),25)-1.5*(np.percentile(df_neg[my_col].dropna(),75)-np.percentile(df_neg[my_col].dropna(),25))
    df_neg[new_col]=df_neg[my_col]
    df_neg.loc[df_neg[new_col] > upper_bound_neg, new_col]  = upper_bound_neg
    df_neg.loc[df_neg[new_col] < lower_bound_neg, new_col]  = lower_bound_neg
    min_sent_neg=df_neg[new_col].min()
    max_sent_neg=df_neg[new_col].max()
    df_neg[new_col] = df_neg[new_col].apply(lambda x: -1 + 1 * (x - min_sent_neg) / (max_sent_neg - min_sent_neg))

    upper_bound_pos=np.percentile(df_pos[my_col].dropna(),75)+1.5*(np.percentile(df_pos[my_col].dropna(),75)-np.percentile(df_pos[my_col].dropna(),25))
    lower_bound_pos=np.percentile(df_pos[my_col].dropna(),25)-1.5*(np.percentile(df_pos[my_col].dropna(),75)-np.percentile(df_pos[my_col].dropna(),25))
    df_pos[new_col]=df_pos[my_col]
    df_pos.loc[df_pos[new_col] > upper_bound_pos, new_col]  = upper_bound_pos
    df_pos.loc[df_pos[new_col] < lower_bound_pos, new_col]  = lower_bound_pos
    min_sent_pos=df_pos[new_col].min()
    max_sent_pos=df_pos[new_col].max()
    df_pos[new_col] = df_pos[new_col].apply(lambda x: 0 + 1 * (x - min_sent_pos) / (max_sent_pos - min_sent_pos))

    df_new = pd.concat([df_neg,df_pos])
    df_new=df_new.sort_values(by='bias').reset_index(drop=True)
    return df_new

# women=["female", "woman", "girl","women", "girls","femininity","feminine","mom","mother"]#"she", "her","hers"
# men=["male", "man", "boy", "men","boys","masculinity","masculine","dad","father"]  #"he", "him","his"

# women=["she", "her","hers"]
# men=["he", "him","his"]


# islam = ["allah", "ramadan", "turban", "emir", "salaam", "sunni", "koran", "imam", "sultan", "prophet", "veil", "ayatollah", "shiite", "mosque", "islam", "sheik", "muslim", "muhammad"]
# christian = ["baptism", "messiah", "catholicism", "resurrection", "christianity", "salvation", "protestant", "gospel", "trinity", "jesus", "christ", "christian", "cross", "catholic", "church"]

# white_names = ["harris", "nelson", "robinson", "thompson", "moore", "wright", "anderson", "clark", "jackson", "taylor", "scott", "davis", "allen", "adams", "lewis", "williams", "jones", "wilson", "martin", "johnson"]
# hispanic_names= ["ruiz", "alvarez", "vargas", "castillo", "gomez", "soto", "gonzalez", "sanchez", "rivera", "mendoza", "martinez", "torres", "rodriguez", "perez", "lopez", "medina", "diaz", "garcia", "castro", "cruz"]

# american_vaccine=['pfizer','pfizerbiontech','pfizervaccine']
# russian_vaccine=['sputnikv']

import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def sexism_finder (df):
    df.fillna(0, inplace=True)
    df1=df[df.bias<0]
    df2=df[df.bias>0]
    misogyny = (df1['bias'].abs()*(df1['freq_pctrank']**1)*df1['hate'].abs()).mean()
    misandry = (df2['bias'].abs()*(df2['freq_pctrank']**1)*df2['hate'].abs()).mean()
    return misandry, misogyny, -misandry+misogyny

def GetTopMostBiasedWords_unicenter(model,adjectives_list_path, c1, pos = ['JJ','JJR','JJS'], verbose = True):
  '''
	modelpath <str> : path to skipgram w2v model
	topk <int> : topk words
	c1 list<str> : list of words for target set 1
	c2 list<str> : list of words for target set 2
	pos list<str> : List of parts of speech we are interested in analysing
	verbose <bool> : True/False
  '''
  def calculateCentroid(model, words, words_sorted):
    embeddings = [np.array(model.wv.__getitem__(w)) for w in words if model.wv.__contains__(w)]
    words_sorted_df=pd.DataFrame(words_sorted)
    weights = [words_sorted_df.loc[words_sorted_df[0]==w,2] for w in words if model.wv.__contains__(w)]
    weights=np.asarray(weights).reshape(len(weights))
    centroid = np.zeros(len(embeddings[0]))
    sum_weights=0
    for i,e in enumerate(embeddings):
        centroid += e*weights[i]
        sum_weights += weights[i]       
    return centroid/sum_weights


  def getCosineDistance(embedding1, embedding2):
    return spatial.distance.cosine(embedding1, embedding2)


    #select the interesting subset of words based on pos
  words_sorted = sorted( [(k,v, model.wv.get_vecattr(k, "count")) for (k,v) in model.wv.key_to_index.items()] ,  key=lambda x: x[1], reverse=False)
  words = [w for w in words_sorted if nltk.pos_tag([w[0]])[0][1] in pos]

  if len(c1) < 1 or len(words) < 1:
    print('[!] Not enough word concepts to perform the experiment')
    return None

  centroid1 = calculateCentroid(model, c1, words_sorted)
  winfo = []
  for i, w in enumerate(words):
    word = w[0]
    freq = w[2]
    rank = w[1]
    pos = nltk.pos_tag([word])[0][1]
    wv = model.wv.__getitem__(word)
    sentNLTK = sid.polarity_scores(word)['compound']
    sentTextBlob = TextBlob(word).sentiment.polarity
    #estimate cosinedistance diff
    bias = getCosineDistance(centroid1, wv)

    winfo.append({'word':word, 'bias':bias, 'freq':freq,
                  'pos':pos, 'wv':wv, 'rank':rank,
                  'sentNLTK':sentNLTK, 'sentTextBlob':sentTextBlob} )

    if(i%100 == 0 and verbose == True):
      print('...'+str(i), end="")

    #Get max and min topk biased words...
  biasc1 = sorted( winfo, key=lambda x:x['bias'], reverse=False )#[:min(len(winfo), topk)]
    #move the ts2 bias to the positive space

  pd.DataFrame(biasc1).to_excel(adjectives_list_path+'.xlsx', index=None)

  return [biasc1]
