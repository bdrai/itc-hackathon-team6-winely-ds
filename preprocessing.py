import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def preprocessing(df):
    sub_df = df.drop(columns=['winery', 'region_1', 'title'])
    dummies = pd.get_dummies(sub_df[["country", "province", "variety"]])
    df_dummies = sub_df.join(dummies).drop(columns=["country", "province", "variety"])
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df_dummies['description'])
    df_description = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df_dummies.index)
    df_final = df_dummies.join(df_description).drop(columns=["description"])
    scaler = StandardScaler()
    scaler.fit(df_final)
    df_final = pd.DataFrame(scaler.transform(df_final), columns=df_final.columns)
    return df_final

