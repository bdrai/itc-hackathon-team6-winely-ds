import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler


def preprocessing(df):
    sub_df = df.drop(columns=['winery', 'region_1', 'title'])
    dummies = pd.get_dummies(sub_df[["country", "province", "variety"]])
    df_dummies = sub_df.join(dummies).drop(columns=["country", "province", "variety"])
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df_dummies['description'])
    df_description = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df_dummies.index)
    df_final = df_dummies.join(df_description).drop(columns=["description"])
    scaler = MinMaxScaler()
    scaler.fit(df_final[["points", "price"]])
    df_final[["points", "price"]] = scaler.transform(df_final[["points", "price"]])
    return df_final

