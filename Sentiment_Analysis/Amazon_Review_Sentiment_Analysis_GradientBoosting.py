
#import libraries
import re
from textblob import TextBlob, Word
import nltk  
import string 
import base64
import itertools
import contractions
import pandas as pd
import streamlit as st
from pickle import dump,load

from nltk import tokenize
from nltk.corpus import stopwords
from unicodedata import normalize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import  TfidfVectorizer
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.ensemble import GradientBoostingClassifier
##############################################################

def add_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/satrapankti/NLP-Natural_Language_Processing/blob/main/Sentiment_Analysis/amazon.png?raw=true");
            background-position: 55% 75%;
            background-size: contain;
            background-repeat: no-repeat
            }}
            </style>
            """,
        unsafe_allow_html=True
    )
add_bg()

st.title("**Sentiment Analysis of Amazon Reviews**")

def review_cleaning(text): 
    stop_words = stopwords.words('english')
    stop_words.extend(["sxsw","@","rt","re","w","u","m","s","sxswi","mention","link","amp","sx","sw","wi","sxs","google","app",
                       "phone","pad","apple","austin","quot","android","ipad","marissa","mayer","social","network","store",
                       "via","popup","called","zlf","zms","quotmajorquot"]) 
    
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore") # Encoding & Decoding Data
    text = contractions.fix(text)                                                      #Contraction Replacement
    text = re.sub("\[.*?\]","", text)                                                  #brackets
    text = re.sub("https?://\S+|www\.\S+", "", text)                                   #links
    text = re.sub("<.*?>+", "", text)                                                  #characters
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)                    #punctuations
    text = re.sub("\n","", text)                                                       #new line
    text = re.sub("\w*\d\w*","", text)                                                 #numbers
    text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",text) if s])           #Split attached Uppercase words
    text = "".join("".join(s)[:2] for _, s in itertools.groupby(text))                 #remove letter repeating twice in continuation
    text = str(text).lower()                                                           #Normalization
    text = " ".join(s for s in str(text).split() if s not in stop_words)               #stopwords
    text = " ". join([w.lemmatize() for w in TextBlob(text).words])                    #Lemmatizaion
    return text

def Polarity(review):    
    return TextBlob(review).sentiment.polarity

def vec(text):
    tf = TfidfVectorizer().fit_transform(text).toarray()
    return pd.DataFrame(tf)


input_review = st.text_input("**:green[Review]**", "Type Here")
product_type = st.selectbox("**:green[Product Type]**",("0","1","2","3","4","5","6","7","8","9"))

#################################################################################################################################################

file = "https://github.com/satrapankti/NLP-Natural_Language_Processing/blob/main/Sentiment_Analysis/Product_details.csv?raw=true"
amazon = pd.read_csv(file) 

az = pd.DataFrame(amazon["Sentiment"])

amazon.drop(["Text_ID"], inplace = True, axis = 1)
amazon["Sentiment"] = amazon["Sentiment"].replace(0,1)
labelencoder = LabelEncoder()
amazon["Sentiment"] = labelencoder.fit_transform(amazon["Sentiment"])

copy = pd.DataFrame(amazon)
amazon.drop(["Sentiment"], inplace = True, axis = 1)
data =  pd.DataFrame({"Product_Description":input_review, "Product_Type":float(product_type)},index = [0])

st.markdown("**:green[User Input parameters]**")
inp = pd.DataFrame(data)
st.write(inp)

##################################################################################################################################################

amazon = pd.concat((amazon, data), axis = 0, ignore_index=True)
amazon["Product_Description"] = amazon["Product_Description"].values.astype(str)
amazon["Product_Description"] = amazon["Product_Description"].apply(review_cleaning)
amazon["Polarity_score"] = amazon["Product_Description"].apply(Polarity)
tf = vec(amazon["Product_Description"])
X = pd.concat((tf, amazon.iloc[:,1:]),axis = 1)
X.columns = X.columns.astype(str)
X_train = X.iloc[:-1]
X_test = X.iloc[-1:] 
Y = az

model = GradientBoostingClassifier()
model.fit(X_train, Y.values.ravel())

#################################################################################################################################################

if st.button("**Predict**"):
    prediction = model.predict(X_test)
    st.subheader("Sentiment")
    st.subheader(prediction[0])


if st.button("**Download ⬇**"):
    prediction = model.predict(X_test)
    data.insert(2, "Sentiment", pd.DataFrame(prediction), True)
    output=pd.concat([copy,data],axis=0, ignore_index=True)
    output.to_csv("Product_details.csv")
    st.markdown("**:red[Downloaded!!]**")
