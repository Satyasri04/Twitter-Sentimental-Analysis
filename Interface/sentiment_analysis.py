import re #for regular expressions
import nltk #for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer 
import gensim 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


pd.set_option("display.max_colwidth",200)
warnings.filterwarnings("ignore",category=DeprecationWarning)

combine= pd.read_csv("train.csv")
def remove_pattern(input_text,pattern):
    r= re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text

combine['tidy_tweet'] = np.vectorize(remove_pattern)(combine['tweet'],"@[\w]*") 
# combine.head()
combine['tidy_tweet'] = combine['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
# combine.head(10)
## Removing short words (a,is,so etc..)
combine['tidy_tweet'] = combine['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #removing words whose length is less than 3
# combine.head()
tokenized_tweet = combine['tidy_tweet'].apply(lambda x:x.split()) #it will split all words by whitespace
# tokenized_tweet.head()

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #it will stemmatized all words in tweet
#now let's combine these tokens back

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #concat all words into one sentence

combine['tidy_tweet'] = tokenized_tweet

all_words = ' '.join([text for text in combine['tidy_tweet']]) 


wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)

# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

#you can see that most words are positive or neutral in above wordcloud.
#now we will plot separate wordclouds for both racist and non-racis/sexist in our data.
## Separate cloud

normal_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==0]])

wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)

# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud,interpolation='bilinear')
# plt.axis('off')
# plt.show()
#racist tweet

negative_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==1]])
wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)

# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud,interpolation='bilinear')
# plt.axis('off')
# plt.show()
## understanding impact of hashtags on tweet sentiment
#collect hashtags

def hashtag_extract(x):
    hashtags=[]
    for i in x: #loop over words contain in tweet
        ht = re.findall(r"#(\w+)",i)
        hashtags.append(ht)
    return hashtags



#extracting hashtags from non racist tweets
ht_regular = hashtag_extract(combine['tidy_tweet'][combine['label']==0])

#extracting hashtags from racist tweets
ht_negative=hashtag_extract(combine['tidy_tweet'][combine['label']==1])

ht_regular = sum(ht_regular,[])
ht_negative = sum(ht_negative,[])

#non-racist tweets

nonracist_tweets = nltk.FreqDist(ht_regular)
df1 = pd.DataFrame({'Hashtag': list(nonracist_tweets.keys()),'Count':list(nonracist_tweets.values())})

#selecting top 20 most frequent hashtags
df1 = df1.nlargest(columns="Count",n=20)
# plt.figure(figsize=(16,5))
ax = sns.barplot(data=df1, x="Hashtag", y="Count")
ax.set(ylabel = "Count")
# plt.show()
#racist tweets

racist_tweets = nltk.FreqDist(ht_negative)
df2 = pd.DataFrame({'Hashtag': list(racist_tweets.keys()),'Count': list(racist_tweets.values())}) #count number of occurrence of particular word

#selecting top 20 frequent  hashtags

df2 = df2.nlargest(columns = "Count",n=20)
# plt.figure(figsize=(16,5))
ax = sns.barplot(data=df2, x="Hashtag",y="Count")
# plt.show()
## Now we will apply assorted techniques like bag of words,TF-IDF for converting data into features

#Bag-of-words

#Each row in matrix M contains the frequency of tokens(words) in the document D(i)

bow_vectorizer = CountVectorizer(max_df=0.90 ,min_df=2 , max_features=1000,stop_words='english')
bow = bow_vectorizer.fit_transform(combine['tidy_tweet']) # tokenize and build vocabulary
# bow.shape
combine=combine.fillna(0) #replace all null values by 0

X_train, X_test, y_train, y_test = train_test_split(bow, combine['label'],
                                                    test_size=0.2, random_state=42)
# print("X_train_shape : ",X_train.shape)
# print("X_test_shape : ",X_test.shape)
# print("y_train_shape : ",y_train.shape)
# print("y_test_shape : ",y_test.shape)



## we will use Multinomial Naive Bayes Classifier
 # Naive Bayes Classifier

model_naive = MultinomialNB().fit(X_train, y_train) 
predicted_naive = model_naive.predict(X_test)


# plt.figure(dpi=600)

mat = confusion_matrix(y_test, predicted_naive);
# sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

# plt.title('Confusion Matrix for Naive Bayes')
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.savefig("confusion_matrix.png")
# plt.show()


score_naive = accuracy_score(predicted_naive, y_test);
# print("Accuracy with Naive-bayes: ",score_naive)

# Function for testing a new text input
# Function for testing a new text input
def test_analizer(input_text):


    # Preprocess the input text
    input_text = remove_pattern(input_text, "@[\w]*")
    input_text = re.sub("[^a-zA-Z#]", " ", input_text)
    input_text = ' '.join([w for w in input_text.split() if len(w) > 3])
    input_text = ' '.join([stemmer.stem(i) for i in input_text.split()])

    # Convert the preprocessed text into Bag-of-Words format
    input_bow = bow_vectorizer.transform([input_text])

    # Make prediction using the trained Naive Bayes model
    prediction = model_naive.predict(input_bow)



    # # Display the result
    if prediction[0] == 0:
       x = "The input text is predicted as Negative."
    if prediction[0] == 1:
       x = "The input text is predicted as Positive."

    return x