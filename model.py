from sklearn.model_selection import train_test_split
import data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


train, test = train_test_split(data.df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names_out())

test_x_vector = tfidf.transform(test_x)

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# svc.score('Test samples', 'True labels')
# print(svc.score(test_x_vector, test_y))
# print(dec_tree.score(test_x_vector, test_y))
# print(gnb.score(test_x_vector.toarray(), test_y))
# print(log_reg.score(test_x_vector, test_y))

# print(f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None))
# print(classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative']))

conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])
# print(conf_mat)

parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc, parameters, cv=5)

svc_grid.fit(train_x_vector, train_y)

# print(svc_grid.predict(tfidf.transform(['I hated this movie and I will never watch it again'])))