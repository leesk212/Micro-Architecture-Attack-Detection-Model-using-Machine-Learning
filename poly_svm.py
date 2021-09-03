from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
import warnings

warnings.filterwarnings('ignore')

raw = pd.read_csv('./train_concetenate_v4.csv', thousands=',')
pd.set_option('display.float_format', '{:.2f}'.format)  # 항상 float 형식으로

x = raw.loc[:, ['IPC', 'INST', 'L3HIT', 'L2HIT', 'Proc Energy (Joules)', 'TEMP', 'PhysIPC', 'INSTnom']]
y = raw.loc[:, ['State']]
X = x.values
y = y.values.reshape(-1, 1)
Y = y.reshape(-1)

data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, random_state=5)

model = svm.SVC(kernel='poly', degree=8, coef0=10, C=5)

# model.fit(data_train, label_train)
# 1
# model = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="poly", degree=8, coef0=10, C=5, gamma=1))
# ])
# 2
# model = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="poly", degree=5, coef0=1, C=5))
# ])
model.fit(data_train, label_train)

predict = model.predict(data_test)
print("학습용 데이터셋 정확도: {:.3f}".format(model.score(data_train, label_train)))
print("검증용 데이터셋 정확도: {:.3f}".format(model.score(data_test, label_test)))
print("리포트 :\n",metrics.classification_report(label_test, predict))
print(confusion_matrix(label_test, predict))
plot_confusion_matrix(model,data_test,label_test,display_labels=['Normal','Flush+Reload','Flush+Flush','Meltdown'])
plt.show()

# metrics.plot_roc_curve(model,data_test,label_test)
# plt.show()



# cm = confusion_matrix(label_test,predict,labels=model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_).plot()
# disp.plot()

# from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)
# precision, recall, _ = precision_recall_curve(label_test, predict)
# disp = PrecisionRecallDisplay(precision=precision, recall=recall)
# disp.plot()
# plt.show()


# scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
# print(scores)

# tuned_parameters = {
#     'C': (np.arange(0.1, 1, 0.1)), 'kernel': ['linear'],
#     'C': (np.arange(0.1, 1, 0.1)), 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05], 'kernel': ['rbf'],
#     'degree': [2, 3, 4], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05], 'C': (np.arange(0.1, 1, 0.1)), 'kernel': ['poly']
# }
# model_svm = GridSearchCV(model, tuned_parameters, cv=10, scoring='accuracy')
# print(model_svm.best_params_)

# Plot non-normalized confusion matrix
