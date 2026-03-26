import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("loading...")

df = pd.read_csv("hinglish.csv", skiprows=1)

print(df.columns)
print(df.shape)

df = df[['message','label']]
df = df.dropna()

print("rows:",len(df))

# just checking labels once
print(df['label'].unique())

# sample dekh liya thoda
print(df.head(5))

X = df["message"]
y = df["label"]

vec = CountVectorizer()
Xv = vec.fit_transform(X)

x1,x2,y1,y2 = train_test_split(Xv,y,test_size=0.2,random_state=42)

print("training...")

m = MultinomialNB()
m.fit(x1,y1)

p = m.predict(x2)

print("acc:",accuracy_score(y2,p))

# thoda check kar liya random samples
i = 0
while i < 3:
    try:
        print("text:",X.iloc[y2.index[i]][:60])
        print("real:",y2.iloc[i],"pred:",p[i])
        print("---")
    except:
        pass
    i += 1

# manual test
while True:
    t = input()
    if t.lower() in ["exit","quit"]:
        break
    if not t.strip():
        continue

    v = vec.transform([t])
    r = m.predict(v)[0]
    prob = m.predict_proba(v)[0]

    print(r,max(prob))