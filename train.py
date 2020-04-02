# encoding=utf-8
from scipy import misc
import numpy as np
def main(src, dst):   # 从src中读取图片的路径和它的标签  dst代表我们将预处理好的图片数据保存到哪里
    with open(src, 'r') as f:  # 读取图片列表
        list = f.readlines()
    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')  # 将图片列表中的每一行拆分成图片名和图片标签
        print(name + ' processed')
        img = misc.imread(name) # 将图片读取出来，存入一个矩阵
        img = img/255 # 原始图片中只有 0 和 255 两种灰度值，我们的代码对图片灰度值除以了 255，将图片矩阵转换成了只包含 0-1 值的矩阵
        img.resize((img.size, 1))  # 为了之后的运算方便，我们将图片存储到一个img.size*1的列向量里面
        data.append(img)
        labels.append(int(label))

    print('write to npy')
    np.save(dst, [data, labels])  # 将训练数据以npy的形式保存到成本地文件
    print('completed')
	
	
if __name__ == '__main__':
    src = r"C:\Users\Jinha\MachineLearning_project\train.txt"
    dst = r"C:\Users\Jinha\MachineLearning_project\train.npy"
    main(src,dst)