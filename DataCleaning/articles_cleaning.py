import os
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('popular')

def preprocess_text(text, flg_stemm=False, flg_lemm=True):
    lst_stopwords = nltk.corpus.stopwords.words("english")

    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    # text = " ".join(lst_text)
    text = lst_text
    return text


# param1:whether the Na value of detail_desc are dropped or not, True or False
# param2:word stemming is used or not, Ture or False
# param3:word lematisation is used or not, True or False
def articles_func(root_dir, is_detailDesc_NAdropped, is_flg_stemmed, is_flg_lemmed):
    df_articles_raw = pd.read_csv(root_dir + '/articles.csv')
    df_articles = df_articles_raw.copy()

    # if drop the rows in which detail_desc is na, or not
    if (is_detailDesc_NAdropped is True):
        df_articles = df_articles.dropna().reset_index()
        df_articles.drop('index', axis=1, inplace=True)
        print("detail_desc dropped!")

    # Clean all text columns
    articles_name = df_articles.filter(regex="name$")
    for i in articles_name.columns:
        new_column_name = 'cleaned_' + i
        if i == 'prod_name':
            df_articles[new_column_name] = df_articles[i].apply(lambda x: x.replace("(1)", '') if "(1)" in x else x)
            df_articles[new_column_name] = df_articles[new_column_name].apply(
                lambda x: preprocess_text(x, flg_stemm=is_flg_stemmed, flg_lemm=is_flg_lemmed, )
            )
        else:
            df_articles[new_column_name] = df_articles[i].apply(
                lambda x: preprocess_text(x, flg_stemm=is_flg_stemmed, flg_lemm=is_flg_lemmed, )
            )

    # Clean Detail Description
    df_articles["cleaned_detail_desc"] = df_articles["detail_desc"].apply(
        lambda x: preprocess_text(x, flg_stemm=is_flg_stemmed, flg_lemm=is_flg_lemmed, )
    )

    df_articles = df_articles.filter(regex='cleaned|no$|code$|id$')
    print("articles cleaning done!")
    return df_articles


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_dir = 'D:/Polimi/M2-S2/'
    df_articles_cleaned = articles_func(root_dir, True, True, True)
    print(df_articles_cleaned.head())