import numpy as np
import pandas as pd
from sklearn import preprocessing
import re
import jpype
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk #stop wordlerden kurtulmak için kullanıldı.
#nltk.download('stopwords')

#jre ve jdk kurulu olması gerekir.

df = pd.read_excel("veriseti.xlsx")
print(df["Duygu"].value_counts())

stop_word_list = nltk.corpus.stopwords.words('turkish')
  
#Veri setinde gürültüye sebep olabilecek kelimeler stop_word listesine eklendi.  
stop_word_list.extend(["bir","kadar","sonra","kere","mi","ye","te","ta","nun","daki","nın","ten"])

#Zemberek kütüphanesini kullanmak için zemberek jar dosyasının konumu verildi
ZEMBEREK_PATH ='C:/Users/Dell/Desktop/beyzaDosyalar/Python/Sentiment Analiz/zemberek-full.jar' 
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
#Lemmatizasyon
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()
TurkishSpellChecker: JClass = JClass('zemberek.normalization.TurkishSpellChecker')
spell_checker: TurkishSpellChecker = TurkishSpellChecker(morphology)

def remove_stopword(tokens):
    
    filtered_tokens = [token for token in tokens if token not in stop_word_list]#stop word'lerden temizlenir.  
    return filtered_tokens
    
def spellChecker(tokens):

    for index,token in enumerate(tokens):    
        #yazım yanlısı varsa ife girer
        if not spell_checker.check(JString(token)):
             if spell_checker.suggestForWord(JString(token)):
                  #kelimenin doğru halini döndürür.
                  tokens[index] = spell_checker.suggestForWord(JString(token))[0]
                  #print((spell_checker.suggestForWord(JString(token))[0]))
    
    corrected = [str(token) for token in tokens]

    return " ".join(corrected)
 
def count_vec(sentences):
        
    count_vectorizer = CountVectorizer()
    
    sparce_matrix = count_vectorizer.fit_transform(sentences)
    
    return sparce_matrix,count_vectorizer.get_feature_names(),count_vectorizer

def tfidf_features(sentence):
    
    Tfidf_Transformer = TfidfTransformer()
    
    Tfidf_Matrix = Tfidf_Transformer.fit_transform(sentence)
    
    return Tfidf_Matrix,Tfidf_Transformer
    
def Lemmatization(sentence):
    
    analysis: java.util.ArrayList = (morphology.analyzeAndDisambiguate(sentence).bestAnalysis())
    token = sentence.split()

    pos=[]
    #kelime köküne inemediği kelimeye UNK demektedir. UNK olan kelimeleri normal hali pos listesine atılır.
    for index,i in enumerate(analysis):   
        if str(i.getLemmas()[0])=="UNK":
            pos.append(token[index])
        else:
            pos.append(str(i.getLemmas()[0]))
        #print("lemma:")
        #print(str(i.getLemmas()[0]))  
    return pos 


y = df["Duygu"].values.tolist()

#Etiketi notr olan veriler çıkarılır.
for index,i in enumerate(y):
    if i==0:
        df.drop(index,axis=0, inplace=True)
    
        
df.reset_index(drop=True,inplace=True)
#print(df["Duygu"].value_counts())

x = df.Tweets
y = df.Duygu


X = []
for i in x:
    #text = i.lstrip(" ") #Cümle basindaki bosluklar kaldirilir.
    text = re.sub('\W+', ' ', str(i))
    text = text.replace('I','ı') #lower() yapıldığı zaman I harfi i olarak çevirildiğinden dolayı replace ile düzeltildi.
    text = text.replace('İ','i') 
    text = text.lower()
    text = " ".join([j for j in text.split() if j not in stop_word_list])
    text = " ".join([i for i in text.split() if len(i)>1])
    text = spellChecker(text.split())
    lemmaList = Lemmatization(text)
    text = " ".join(lemmaList)
    X.append(text)
    

Tfidf_Vector = TfidfVectorizer(max_features = 3000)

Tfidf_Matrix = Tfidf_Vector.fit_transform(X)

X_train = Tfidf_Matrix.toarray()

features = Tfidf_Vector.get_feature_names()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33,random_state = 42)

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Xgboost score: "+str(accuracy))


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(x_train, y_train)

y_pred2 = clf.predict(x_test)

print("Multi score:")
print(accuracy_score(y_test,y_pred2))
