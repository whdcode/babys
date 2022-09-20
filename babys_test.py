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
        return array(returnvec)

    def trainNB0(self, trainmatrix, traincategory):  # 数据集矩阵trainmatrix是已经被通过setofwords2vec被二进制向量化
        """朴素贝叶斯分类器训练函数"""
        numtraindocu = len(trainmatrix)
        numword = len(trainmatrix[0])
        P1busive = sum(traincategory) / float(numtraindocu)  # 通过计算分类标签列表中的1的数目求得类别一的概率，二分类有PB= 1 - PA
        # 构造与词条向量等长的全零向量记录各个词条数量
        P0num = ones(numword)
        P1num = ones(numword)
        # 累加并保存各个类别中出现词条总数
        P0Denom = 2.0  # 初始为2，避免后续因数值过小相乘对结果影响
        P1Denom = 2.0
        for i in range(numtraindocu):
            if traincategory[i] == 1:
                P1num += trainmatrix[i]
                P1Denom += sum(trainmatrix[i])
            else:
                P0num += trainmatrix[i]
                P0Denom += sum(trainmatrix[i])

        P1vec = array(log(P1num / P1Denom))  # log() 返回 x 的自然对数,且后续需要进行数组对应元素相乘，故转化为数组
        P0vec = array(log(P0num / P0Denom))

        return P0vec, P1vec, P1busive

    def classifyNB(self, vec2classify, p0vec, p1vec, pclass1):
        """进行分类操作"""
        p1 = sum(vec2classify * p1vec) + log(pclass1)  # vec2classify为需要分类的词条向量对象，已将其2进制向量化
        p0 = sum(vec2classify * p0vec) + log(1 - pclass1)
        if p1 > p0:
            return "1"
        else:
            return "0"

    def get_words(self):
        """获得一句测试用例文字"""
        try:
            words = input("请输入你对斑点狗的评价？(English)")
        except ValueError:
            pass
        else:
            word_list = words.split(' ')
            return word_list

    def testingNB(self):
        """测试分类效果"""
        listoposts, listclasss = self.postingList, self.classVec
        myvocablist = self.createVocablist(listoposts)
        trainmat = []
        for postindoc in listoposts:
            trainmat.append(self.setofwords2vec(myvocablist, postindoc))
        p0v, p1v, p1b = self.trainNB0(trainmat, listclasss)
        while True:
            print("（press 'exit' to exit）.")
            testentry = self.get_words()
            if testentry[0] == 'exit':
                break
            else:
                words2vec = self.setofwords2vec(myvocablist, testentry)
                class_result = self.classifyNB(words2vec, p0v, p1v, p1b)
                print(f"testentry ' classified as: '{class_result}'.")

    # 朴素贝叶斯词袋模型
    def bagofword2vecmn(self, vocablist, inputset):
        """基于词袋模型的向量化"""
        returnvec = [0] * len(vocablist)
        for word in inputset:
            if word in vocablist:
                returnvec[vocablist.index(word)] += 1
        return returnvec

    