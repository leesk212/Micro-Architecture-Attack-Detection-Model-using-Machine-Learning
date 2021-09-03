
# **Microarchitecture Attack Detection Model using Machine Learning**

# Overview

![2021-09-03_18-40-26](https://user-images.githubusercontent.com/67637935/131985282-2bd4c125-22f3-4b02-b852-005c5087df40.png)


## **연구 배경**

공격자가 피해자 컴퓨터의 비밀 정보를 탈취하는 방법에는 다양한 방법이 있다. 그 중 인텔  CPU의 취약점을 악용한 공격들이 치명적이며 이를 방어 및 예방하기 위해서는 공격들의 기저가 되는 Micro-Architecture Attack (Flush+Reload, Flush+Flush, Meltdown)을 탐지할 할 수 있어야 한다. 인텔 CPU의 취약점을 목표로 만들어진 공격들이 피해자 컴퓨터 에서 실행될 때 HPC(Hardware performance counter) 상태에 영향을 주며 이는 PCM (Processesor Counter Monitor)을 통해 HPC상태 변화를 확인할 수 있다. PCM으로부터 추출할 수 있고, 기존의 공격 탐지 논문에서 Feature로 사용한 Cache, IPC뿐만이 아니라, Temp, Power등 추가적인 Feature들로 다항식 커널 SVM(Support Vector Machine) 알고리즘을 활용해 기계 학습을 진행한다. 학습 된 모델로 피해자의 컴퓨터가 어떤 Micro-Architecture Attack을 받고 있는지와 평상시 상태를 구분하는 과정을 통해 공격을 예방할 수 있는 모델을 만들어 공격 탐지를 진행한다.

## **연구 방법**

공격을 실행시키기 위해, 피해자가 되는 컴퓨터를 선정 후, 선정된 컴퓨터 내에서 PCM을 실행시키고 Micro-architecture Attack을 실행시킨다. PCM이 보여주는 상태 변화들을 확인하는 과정을 csv파일로 추출 후 전처리 작업을 진행하였다. 기계 학습을 위해 sklearn의 SVM model API와 poly feature를 사용하였고  만들어진 모델로 테스트하였다.
선정된 컴퓨터는 i5-7400이고 운영체제는 ubuntu 14.04, kernel은 4.10.10을 사용하였다. 이후 PCM을 github를 통해 설치하였고, Mastik에 있는 Flush+Reload, Flush+Flush PoC(Proof of Code) 공격 코드를 변형해 공격자가 멈출 때까지 공격이 실행되도록 코드를 재구성하였다. Meltdown 공격의 경우 Meltdown 저자가 구성한 PoC를 github에서 받고 이전 공격들과 동일하게 변형을 진행해 공격자가 멈출 때까지 Meltdown 공격이 실행되도록 코드를 재구성하였다.

### **1.**   **Dataset/Cleaning**
그림 1(좌)와 같이 총 30분간의 PCM을 실행시켰으며 0분~10분: normal state, 10~15분:  Flush+Reload 공격 state, 15분~20분: Flush+Flush 공격 state, 20~30분: Meltdown 공격 state로 시분할을 진행한 뒤, 30분 동안 실행시킨 PCM의 csv파일을 얻었다. csv파일에 PCM을 실행시킨 시작 시간이 Time label로 적힘으로 csv파일의 첫번째 행에 각 시간별 state를 적어 준 뒤, 추가적인 전처리를 진행하였다.  
![image](https://user-images.githubusercontent.com/67637935/116090757-2a264100-a6df-11eb-9b3b-08e03e7f6050.png)![image](https://user-images.githubusercontent.com/67637935/116090772-2e525e80-a6df-11eb-811c-e809cbe7fe8f.png)

그림 1. (좌)PCM 실행 ,(우)공격 파일 목록  
![image](https://user-images.githubusercontent.com/67637935/116090780-30b4b880-a6df-11eb-87f8-bb8fdd37a926.png)  

그림 2. PCM으로부터 산출된 csv파일에 시간별 State 추가한 csv파일 일부, 첫번째 행에 시간별 state를 수기로 작성함.

기존의 PCM에서 산출되는 column의 수는 128개이며, column들 중 각각의 Micro-Architecture Attack 상태와 normal 상태를 구분할 수 있는 Feature 찾는 과정을 진행하였다. 중복되는 값을 가진 column들을 제거하고, CPU의 특성으로 PCM이 산출해 내지 못해  column의 값이 nan값이거나, 변화가 없는 column들은 제거해주었다. 최종적으로 State에 따른 값의 변화가 있는 column은 'INST', 'ACYC', 'PhysIPC', 'INSTnom','Proc Energy (Joules)', 'IPC', 'FREQ', 'AFREQ', 'L3MISS', 'L2MISS', 'L3HIT', 'L2HIT', 'L3MPI', 'L2MPI', 'READ', 'TEMP' 16가지로 선정하였다.

### **2.**   **Feature Manipulation**
16가지의 column들을 state에따른 값의 변화 확인을 하여 Feature로 선정하고자 하였다. 직관적인 방법을 위해 시간에 따른 값들의 변화를 시각적으로 볼 수 있는 코드가 필요하였고, matplotlib와 pandas, numpy를 통해 이를 진행하였다. 그림 3과 같이 state에 따른 큰 변화의 차이가 없는 column들은 Feature 후보에서 제외 되었으며, 그림 4와 같이 확실한 변화가 보이는 column들은 Feature로 선정하였다. 'IPC', 'INST', 'L3HIT', 'L2HIT', 'Proc Energy (Joules)', 'TEMP', 'PhysIPC', 'INSTnom' 8가지의 column을 Feature로 선정하였다. 8개의 Feature값이 정규화가 필요할 정도로 값의 변화가 크지 않았기에 기존 Feature 값 그대로 사용하였다.  
![image](https://user-images.githubusercontent.com/67637935/116090853-44f8b580-a6df-11eb-87f1-10beef62949c.png)
![image](https://user-images.githubusercontent.com/67637935/116090862-475b0f80-a6df-11eb-9067-93bbf066a17c.png)
![image](https://user-images.githubusercontent.com/67637935/116090872-4924d300-a6df-11eb-80e1-1ab1b5b32db6.png)
![image](https://user-images.githubusercontent.com/67637935/116090882-4a560000-a6df-11eb-9333-707d99d258c2.png)
![image](https://user-images.githubusercontent.com/67637935/116090888-4c1fc380-a6df-11eb-8436-93fb712b758a.png)
![image](https://user-images.githubusercontent.com/67637935/116090893-4de98700-a6df-11eb-9c3e-b28590f184c2.png)
![image](https://user-images.githubusercontent.com/67637935/116090901-4fb34a80-a6df-11eb-99fd-549628fd23c1.png)
![image](https://user-images.githubusercontent.com/67637935/116090909-517d0e00-a6df-11eb-9ff2-1e851068313e.png)

그림 3. 16가지의 column중 상태별 변화가 부족한 column들의 시각적 표현  
![image](https://user-images.githubusercontent.com/67637935/116090923-5346d180-a6df-11eb-8917-7fe9d0a61c27.png)
![image](https://user-images.githubusercontent.com/67637935/116090936-56da5880-a6df-11eb-94c2-68e816e9d9e6.png)
![image](https://user-images.githubusercontent.com/67637935/116090944-58a41c00-a6df-11eb-8ce0-66faee4d9b23.png)
![image](https://user-images.githubusercontent.com/67637935/116090956-5b067600-a6df-11eb-9f26-1c950220ef5c.png)
![image](https://user-images.githubusercontent.com/67637935/116090964-5cd03980-a6df-11eb-8785-f198f667109e.png)
![image](https://user-images.githubusercontent.com/67637935/116090971-5e99fd00-a6df-11eb-9abe-92bc5a8ba2c9.png)
![image](https://user-images.githubusercontent.com/67637935/116090979-60fc5700-a6df-11eb-8a74-ac32ed72dc37.png)
![image](https://user-images.githubusercontent.com/67637935/116090992-648fde00-a6df-11eb-94b1-4fe7619e319c.png)

그림 4. 16가지의 column 중 Feature로 사용한 column들의 시각적 표현

### **3.**   **SVM - Classification**
전처리가 끝난 데이터를 다항식 커널 SVM Classification을 활용한 기계학습을 진행하였다. 모델을 만들기 위해 Hands-On Machine Learning 책의 moons 데이터셋을 SVC 커널 트릭(kernel trick)으로 구현한 코드를 참고하였다. 전처리가 완료된 데이터들을 sklearn의 train_test_split API를 사용하여 train data set과 test data set을 8:2의 비율로 분리한 후 train data를 model의 input값으로 주어 학습을 진행하였다. 그림 5. 참조
```python
data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, random_state=11)

model = svm.SVC(kernel='poly', degree=1, coef0=1, C=5)

model.fit(data_train, label_train)

```
그림 5. 다항 커널 SVM 기계학습 코드

이후 Feature의 개수가 8개임으로 최적의 성능을 위한 파라미터를 찾는 과정을 진행하였다.  SVC의 degree 인자 값과 coef0의 값을 증가시키면서 metrics의 classification_report API로부터 나오는 값들과 Confusion Matrix의 값을 확인하는 코드를 구성하고 결과를 비교하며 연구를 진행하였다. 그림 6참조
```python
predict = model.predict(data_test)
print("학습용 데이터셋 정확도: {:.3f}".format(model.score(data_train, label_train)))
print("검증용 데이터셋 정확도: {:.3f}".format(model.score(data_test, label_test)))
print("리포트 :\n",metrics.classification_report(label_test, predict)) 
plot_confusion_matrix(model,data_test,label_test,display_labels=['Normal','Flush+Reload','Flush+Flush','Meltdown'])
plt.show()

```
그림 6. 학습 모델 평가 코드

파라미터 값의 변화 이외에 SVM은 특성의 스케일에 민감하기에, 특성의 스케일을 조정하는 sklearn의 StandardScaler API를 사용하여 결정 경계의 상태를 더욱 좋아지도록 모델을 추가 구현하여 이전 모델들과 성능을 비교하며 연구를 진행하였다. 그림 7. 참조
```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=5, coef0=1, C=5))
])

```
그림 7. StandardScaler API를 사용해 추가  구현한 모델 코드

### **4.**   **Evaluation**
구성한 SVM-Classification 모델의 목적은 4개의 state를 8개의 feature의 값을 통해 학습한 뒤, test를 위해 8개의 feature들의 값을 input으로 넣어, 어떤 state인지 확인할 수 있는 기계학습을 진행하는 것이다.
Classification 모델이기에, 평가하는 요소는 Confusion Matrix로부터 나올 수 있는 평가를 활용하였다. 평가 API는 크게 세 가지를 사용하였다. 1) skitlearn의 score API, 2) skitlearn의 metircs method의 classification_report API, 3) skitlearn의 metircs method의 plot_confusion_martix API 이다. 첫번째 API를 통해서 만들어진 모델들에게 train data set와 test data set을 주어 scoring을 진행해 Acc(Accuracy)를 확인하는 과정을 진행하였다. 그림 8의 좌측 그림의 첫번째, 두번째 줄의 값이다. 두번째 API를 통해서 state에 따른 전체적인 precision, recall, f1-score와 support(평가에 사용된 state data 갯수)를 보여준다. 이 API를 사용하기 위해서는 test data set를 predict하는 method가 선행 되어야한다. 그림 6. 참조. API의 결과는 그림 8의 좌측 그림의 세번째 줄부터 나온 결과 값들이며 각각의 state에 맞게 precision 값과, recall 값과 f1-score, support값을 알 수 있고 최종적으로 총 support의 갯수와 Acc를 알 수 있다. Acc를 통해서만 model의 성능을 확인하기엔 정보가 부족할 수 있으니, 각각의 state에 맞게 precision, recall, f1-score의 값을 확인할 수 있어 용의했다. 마지막으로 평가에 사용한 API는 시각적으로 평가를 볼 수 있는 plot_confusuin_matrix API를 사용하였다.
svm 모델의 파라미터들의 값을 변경하면서 연구를 진행하였고, 그 결과는 다음의 그림 8로 정리하였다. degree가 증가할 수록 구분할 수 있는 차원이 늘어나기에 더 높은 정확도를 볼 수 있었다(그림8 -A, B 비교). 하지만 한계가 있었기에 kernel 함수의 독립적인 구간을 설정하는 coef0를 적절히 설정해주었고 그 평가 결과(그림 8 -C), 적절히 coef0를 설정한 모델이 더욱 높은 정확도를 보여주었다. 이후 StandardScaler API를 적용한 모델과 아닌 모델을 degree 인자 값을 동일하게 1로 설정 후 비교 진행하였다. SVM모델의 결정 경계 상태를 더욱 좋게 해주는 StandardScaler API를 사용한 모델에서 더 정확도가 높았음을 확인할 수 있었다.(그림 8 -A, D 비교). 그 이후 부터는 StandardScaler API를 사용한 모델에서 degree와 coef0 파라미터들을 증가시키면서 비교 진행을 하였다. degree가 늘어나면 더 정확도가 증가함을 확인할 수 있었지만(그림 8 -E,F 비교) coef0의 파라미터의 값을 주었을 때는 StandardScaler API를 사용하지 않았을 떄 와 달리 큰 정확도의 향상은 볼 수 없었다.  
![image](https://user-images.githubusercontent.com/67637935/116091171-8db06e80-a6df-11eb-8dfe-d00997fb0cc4.png)

A.	degree 1의 결과  
![image](https://user-images.githubusercontent.com/67637935/116091181-8f7a3200-a6df-11eb-927f-224e94a79f53.png)

B.	degree 8의 결과  
![image](https://user-images.githubusercontent.com/67637935/116091197-92752280-a6df-11eb-97f4-b28bd6c2a0e9.png)

C.	degree 8, coef0=10의 결과  
![image](https://user-images.githubusercontent.com/67637935/116091209-94d77c80-a6df-11eb-818e-ba76150a0c09.png)

D.	StandardScalar() API를 추가로 사용한 모델의 결과 (degree = 1)  
![image](https://user-images.githubusercontent.com/67637935/116091221-9739d680-a6df-11eb-9411-353ab339ce1b.png)

E.	StandardScalar() API를 추가로 사용한 모델의 결과 (degree = 8)  
![image](https://user-images.githubusercontent.com/67637935/116091235-99039a00-a6df-11eb-86a2-30e58cc4956e.png)

F. StandardScalar() API를 사용한 모델 결과(degree = 8, coef0=10)  

그림 8. 모델 테스트 결과 모음

## **연구 결과 및 고찰**

Evaluation을 토대로 SVM model의 degree 파라미터의 값을 1로 주었을 때 Acc는 0.932가 나왔다. 그리고 추가적으로 coef0 파라미터와 StandardScalar API를 주고,

**참고 문헌**
[1] Y. Dong et al., “Driver Inattention Monitoring System for Intelligent Vehicles: A Review,” *IEEE Intelligent Transportation Systems*., vol.12, pp.596-614, 2011.
[2] M. Yeo et al., “Can SVM be used for automatic EEG detection of drowsiness during car driving,” *Safety Science*., vol.47,
