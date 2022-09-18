"""
date:2022年9月18日16点37分
name:朴素贝叶斯（基于贝努利模型）
"""
from numpy import *


class Byes:
    """朴素贝叶斯算法"""

    def __init__(self):
        """初始化文档数据集列表和类别标签列表"""
        self.postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        self.classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not

    def createVocablist(self, dataset):
        """创建不重复单词列表"""
        vocabset = set([])  # 创建一个空集合
        for document in dataset:
            vocabset = vocabset | set(document)  # 不断并入每一文档数据集中的不重复单词
        return list(vocabset)

    def setofwords2vec(self, vocablist, inputset):
        """生成词条向量"""
        returnvec = [0] * len(vocablist)  # 定义一个全为0的和单词表一样长度的向量列表
        for word in inputset:
            if word in vocablist:
                returnvec[vocablist.index(word)] = 1  # 若单词在单词表上出现过，则根据单词表对应的索引，在自定义向量列表的对应索引元素赋1
            else:
                print(f"the word : {word} is not in my vocabulary!")
        return returnvec

    def trainNB0(self, trainmatrix, traincategory):  # 数据集矩阵trainmatrix是已经被通过setofwords2vec被二进制向量化
        """朴素贝叶斯分类器训练"""
        numtraindocu = len(trainmatrix)
        numword = len(trainmatrix[0])
        P1busive = sum(traincategory) / float(numtraindocu)  # 通过计算分类标签列表中的1的数目求得类别一的概率，二分类有PB= 1 - PA
        # 构造与词条向量等长的全零向量记录各个词条数量
        P0num = zeros(numword)
        P1num = zeros(numword)
        # 累加并保存各个类别中出现词条总数
        P0Denom = 0.0
        P1Denom = 0.0
        for i in range(numtraindocu):
            if traincategory[i] == 1:
                P1num += trainmatrix[i]
                P1Denom += sum(trainmatrix[i])
            else:
                P0num += trainmatrix[i]
                P0Denom += sum(trainmatrix[i])

        P1vec = log(P1num / P1Denom)  # log() 返回 x 的自然对数
        P0vec = log(P0num / P0Denom)

        return P0vec, P1vec, P1busive
