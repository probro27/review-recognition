import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

df_reviews = pd.read_csv('IMDB Dataset.csv')

df_positive = df_reviews[df_reviews['sentiment'] == 'positive'][:9000]
df_negative = df_reviews[df_reviews['sentiment'] == 'negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

rus = RandomUnderSampler(random_state=0)

df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_imb[['review']], df_review_imb[['sentiment']])

# print(df_review_bal.value_counts('sentiment'))