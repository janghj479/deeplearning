# coding: utf-8
import sys, os # 시스템(system)과 관련된 기능이 담겨 있는 sys 모듈 / 운영체제와 관련있는 os 모듈 임포트
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np # 행렬이나 대규모 다차원 배열을 쉽게 처리 할 수 있도록 지원하는 numpy를 np로 임포트
import pickle # 프로그램 실행 중에 특정 객체를 파일로 저장하는 기능을 임포트
from dataset.mnist import load_mnist # dataset.mnist에서 load_mnist 임포트
from common.functions import sigmoid, softmax # common.functions에서sigmoid, softmax 임포트


def get_data(): #데이터를 가져오는 함수 
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 읽은 MNIST데이터를 (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형식으로 반환
    """
    normalize : 
               True 설정시 / 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화 
               False 설정시 /  입력 이미지의 픽셀은 원래 값 그대로 0 ~ 255 사이의 값을 유지               
               
    flatten :  
               True 설정시 / 입력 이미지를 평탄하게 즉 1차원 배열로 설정 > 784개의 원소로 이루어진 1차원 배열로 저장 
               False 설정시 / 입력 이미지 배열 그대로 사용 > 1 x 28 x 28의 3차원 배열로 저장             
               
    one_hot_label : 
               True 설정시 / 원핫인코딩 형태로 저장할지를 설정 > 정답을 뜻하는 원소만 1이고 나머지는 0인 배열
               False 설정시 / 숫자 형태의 배열로 저장    
               
    * 정규화(normalization) : 데이터를 특정 범위로 변환.
    ** 전처리(pre-processing) : 입력 데이터에 특정 변환을 가하는 것.
               
    """
    return x_test, t_test # 값을 반환
    
    
def init_network():
    with open("sample_weight.pkl", 'rb') as f: # with as 구문을 사용하여 sample_weight.pkl를 이진 파일로 읽는다. f로 부른다. 
        network = pickle.load(f) # sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 network에 저장.
    return network # 값을 반환


def predict(network, x): # 예측값을 출력하는 함수
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # network에서 가져온 가중치를 각 변수마다 할당 
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # network에서 가져온 바이어스를 각 변수마다 할당 

    a1 = np.dot(x, W1) + b1 # x와 w1을 내적한 값에 b1을 더하여 a1에 저장
    z1 = sigmoid(a1) # softmax 함수를 통과한 a1을 z1에 저장
    a2 = np.dot(z1, W2) + b2 # z1과 w2을 내적한 값에 b2을 더하여 a2에 저장
    z2 = sigmoid(a2) # softmax 함수를 통과한 a2를 z2에 저장
    a3 = np.dot(z2, W3) + b3 # z2와 w3을 내적한 값에 b3을 더하여 a3에 저장
    y = softmax(a3) # softmax 함수를 통과한 a3을 y에 저장

    return y # 값을 반환

# 정확도(accuracy : 분류가 얼마나 올바른가) 평가
x, t = get_data() # MNIST 데이터셋 얻기
network = init_network() # 네트워크 생성
accuracy_cnt = 0 # 변수 0으로 초기화

for i in range(len(x)): #for문을 돌면서 x 에 저장된 이미지 데이터를 1장씩 꺼냄 
    y = predict(network, x[i]) # 1개씩 꺼낸 데이터를 predict()함수로 분류 
    # predict() 함수는 각 레이블의 확률을 넘파이 배열로 반환
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다. >> 예측결과
    if p == t[i]: #신경망이 예측한 답변과 정답레이블을 비교
        accuracy_cnt += 1  # 비교해서 맞힌 숫자를 셈

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) # 맞힌 숫자/ 전체 이미지 숫자 >> 정확도를 구한다. 