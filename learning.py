from sklearn import svm, metrics
import pandas as pd
# XOR의 계산 결과 데이터
xor_input = [
    #P,Q,result
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

# 학습을 위해 데이터와 레이블 분리하기
### 왜 분리하느냐? fit이라는 메서드의 매개변수에는 훈련용,테스트용 두개의 변수가 필요하기 때문에 나눈다.
#- 방법 1
    # xor_data = []
    # xor_label = []
    # for row in xor_input:
    #     p = row[0]
    #     q = row[1]
    #     r = row[2]
    #     xor_data.append([p,q])
    #     xor_label.append(r)
#- 방법 2
xor_df = pd.DataFrame(xor_input) # pandas 패키지에는 R같이 데이터 프레임을 만들어주는 메서드가 있다.
xor_data = xor_df.ix[:,0:1] # 데이터
xor_label = xor_df.ix[:,2] # 레이블

# 데이터 학습시키기
clf = svm.SVC()
clf.fit(xor_data,xor_label) # fit => 괄호 안에 있는 데이터들을 학습시키는 함수 -> fit( 학습할 데이터, 레이블 배열 )

# 데이터 예측하기
pre = clf.predict(xor_data) # > 예측하고 싶은 데이터 배열을 매개변수로 넣는다.
# print("예측결과 :",pre) # 예측결과 : [0,1,1,0]


# # 결과 확인하기
# ok = 0;total = 0
# for idx,answer in enumerate(label):
#     p = pre[idx]
#     if p == answer: ok += 1
#     total += 1
# print("정답률:",ok,"/",total,"=",ok/total)   # 정답률 = 1.0

ac_score = metrics.accuracy_score(xor_label,pre)
print("정답률 =",ac_score)  # 정답률 = 1.0