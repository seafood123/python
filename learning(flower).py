from sklearn import svm,metrics
import random,re

# 붓꽃의 csv파일 읽어 들이기
csv = []
with open("iris.csv","r",encoding="utf-8") as fp:
    # 한 줄씩 읽어 들이기
    for line in fp:
        line = line.strip() # 줄 바꿈 제거
        cols = line.split(",") # 쉼표로 자르기 (CSV파일은 쉼표로 데이터를 쪼개놓음)
        # 문자열 데이터를 숫자로 변환하기
        fn = lambda n:float(n) if re.match(r'^[0-9\.]+$',n) else n  # 해당 셀의 내용이 숫자면 float로 실수로 변환한다.
        cols = list(map(fn,cols)) # 리스트에 적용하는 map() 함수로 리스트 내부의 모든 값을 변환한다.
        csv.append(cols)

# 맨 앞 줄의 헤더 제거
del csv[0]

# 데이터 섞기
random.shuffle(csv)

# 학습 전용 데이터와 테스트 전용 데이터 분할하기(2:1비율)
total_len = len(csv)
train_len = int(total_len *2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(total_len):
    data = csv[i][0:4] # 0,1,2,3열 저장
    label = csv[i][4] # 4열 저장
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

# 데이터를 학습시키고 예측하기
clf = svm.SVC()
clf.fit(train_data,train_label)
pre = clf.predict(test_data)

# 정답률 구하기
ac_score = metrics.accuracy_score(test_label,pre)
print("정답률 =",ac_score)