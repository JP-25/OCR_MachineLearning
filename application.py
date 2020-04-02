from scipy import misc
# from PIL import Image
import numpy as np
import os
import cv2

def modify_image (src):
    def Dilation (image):
	    kernel = np.ones((3,3), np.uint8)
	    new_img = cv2.dilate(image, kernel, iterations=1)
	    return new_img
    
    def Erosion (image):
        kernel = np.ones((3,3), np.uint8)
        new_img = cv2.erode(image, kernel, iterations=1)
        return new_img
    img = cv2.imread(src)
    cv2.imshow('sample image',img)
    cv2.waitKey(0)
    gray = cv2.imread(src, 0)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # image = Image.open(src)
    # gray = image.convert('L')
    height, width = gray.shape
    # height, width = gray.size
    flag = True
    h1, w1 = gray.shape
    if h1 <= w1 :
        w1 = h1
    elif w1 <= h1 :
        h1 = w1
    gray = cv2.resize(gray, (h1,w1))
    # cv2.imshow('image', gray)
    # cv2.waitKey(0)
    while height >= 34 and width >= 34:
        gray = cv2.resize(gray, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LANCZOS4)
        height, width = gray.shape
        # if height/h1 >= 0.23 :
        #     gray = Dilation(gray)
            # print('dilation successfully') #debugging numbers of Dilations
        # cv2.imshow('image', gray)
        # cv2.waitKey(0)
        if flag:
            gray = Dilation(gray)
            flag = False
        else:
            flag = True
    gray = cv2.resize(gray, (17,17))
    # gray.thumbnail((17,17))
    gray = gray / 255
    # gray[gray > 0 ] = 1
    gray.resize((gray.size,1))
    # print(gray.shape)
    return gray	
	
class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)  # 使用随机数初始化参数，请暂时忽略这里为什么多了np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)  # 使用随机数初始化参数
        # self.lr = 0  # 先将学习速率初始化为0，最后统一设置学习速率
	
    def parameters_load (self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        self.x = x  # 把中间结果保存下来
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递
		

class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y
		

def main():
    print ("loading...")
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers[0].parameters_load(np.load('weights.npy', allow_pickle=True), np.load('biases.npy', allow_pickle=True))
    inner_layers.append(Sigmoid())
    print ("tranning completed")
    
    while True:
        print ("PATH of image name: ")
        path = input()
        if (path == "exit"):
            break

        process = modify_image(path)
        x = []
        # 26 letters, make 26 times
        for i in range (1, 26):
            x.append(process)
        for layer in inner_layers:
            x = layer.forward(x)
        
        # print(inner_layers[1].y)
        # print(np.argmax(inner_layers[1].y))


        # ASCII, plus 65 get English letters
        print(chr(np.argmax(inner_layers[1].y) + 65))
		
		
if __name__ == '__main__':
    main()