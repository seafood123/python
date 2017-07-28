from sklearn import svm
from sklearn.externals import joblib
import json

# 각 언어의 출현 빈도 데이터(json) 읽어 들이기
with open("./ex/ch4/lang/freq.json","r",encoding="utf-8") as fp:
    d = json.load(fp)
    data = d[0]

# 데이터 학습하기
clf = svm.SVC()
clf.fit(data['freqs'],data['labels'])

# 학습데이터 저장하기
joblib.dump(clf,"./ex/ch4/lang/freq.pkl")
print("ok")