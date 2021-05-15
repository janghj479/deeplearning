
import sys, os
sys.path.append("/home/janghj479/deep-learning-from-scratch/") # 부모 디렉터리의 파일을 가져옴

from dataset.mnist import load_mnist

# 처음에 불러오려면 오래걸림
(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape) # (60000, 784) 
print(t_train.shape) # (60000, )
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000, )