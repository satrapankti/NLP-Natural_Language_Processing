
#import libraries
import re
from textblob import TextBlob, Word
import nltk  
import string 
import base64
import itertools
import contractions
import pandas as pd
from pickle import dump,load

from nltk import tokenize
from nltk.corpus import stopwords
from unicodedata import normalize


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import  TfidfVectorizer
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

import streamlit as st 
from sklearn.ensemble import GradientBoostingClassifier
########################### 


def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/satrapankti/NLP-Natural_Language_Processing/blob/main/Sentiment_Analysis/amazon.png");
            background-position: 55% 75%;
            background-size: contain;
            background-repeat: no-repeat
            }}
            </style>
            """,
        unsafe_allow_html=True
    )
add_bg_from_local() 


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


input_review = st.text_area("**:green[Review]**", "Type Here")
product_type = st.selectbox("**:green[Product Type]**",("0","1","2","3","4","5","6","7","8","9"))

#######################################################################################

file = "https://github.com/satrapankti/NLP-Natural_Language_Processing/blob/main/Sentiment_Analysis/Product_details.csv?raw=true"
az = pd.read_csv(file) 
amazon = pd.DataFrame(az["Product_Description"])
amazon.loc[len(amazon)] = [input_review]
amazon["Product_Description"] = amazon["Product_Description"].values.astype(str)
amazon["Product_Description"] = amazon["Product_Description"].apply(review_cleaning)

result = amazon["Product_Description"].iloc[-1:].to_string(index = False)
tfi = vec(amazon["Product_Description"])
tfi.columns = tfi.columns.astype(str)
tf = tfi.iloc[-1:]
polar = Polarity(result)
tf["Product_Type"] = product_type
tf["Polarity_score"] = polar

########################################################################################

def user_input_features_text():
    tf.columns = tf.columns.astype(str)
    return tf 


df = user_input_features_text()
st.markdown("**:green[User Input parameters]**")
inp = pd.DataFrame({"Review":result, "Product Type":product_type, "Polarity Score":round(polar,4)},index = [0])
st.write(inp)

##################################################################################################################

sent = pd.read_csv(file) 
sent.drop(["Text_ID"],inplace=True,axis = 1)
sent["Sentiment"] = sent["Sentiment"].replace(0,1)
labelencoder = LabelEncoder()
sent["Sentiment"] = labelencoder.fit_transform(sent["Sentiment"])
sent["Product_Description"] = sent["Product_Description"].values.astype(str)
sent["Product_Description"] = sent["Product_Description"].apply(review_cleaning)
sent["Polarity_score"] = sent["Product_Description"].apply(Polarity)
tf = vec(sent["Product_Description"])
X = pd.concat((tf,sent["Product_Type"],sent["Polarity_score"]),axis = 1)
X.columns = X.columns.astype(str)
Y = sent["Sentiment"]
model = GradientBoostingClassifier()
model.fit(X,Y)

###################################################################################################################

if st.button("**Predict**"):
    prediction = model.predict(df)
    st.subheader("Sentiment")
    st.subheader(prediction[0])
    st.subheader("Predicted Result")
    if prediction ==  0: 
        st.write("**:red[Negative]**")
    elif prediction == 1:
        st.write("**:red[Neutral]**") 
    else:
        st.write("**:red[Positive]**")

if st.button("**Download ⬇**"):
    prediction = model.predict(df)
    output=pd.concat([df,pd.DataFrame(prediction)],axis=1)
    output.to_csv("prediction.csv")
