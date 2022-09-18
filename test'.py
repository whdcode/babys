from babys_test import *
postingList_ = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec_ = [0, 1, 0, 1, 0, 1]
# 定义一个实例
by = Byes()
my_vocab = by.createVocablist(by.postingList)   # 得到不重复单词表
print(my_vocab)
# 生成文档数据集中第一个文档的2进制词条向量
first_vocab2Vec = by.setofwords2vec(my_vocab, by.postingList[3])
print(first_vocab2Vec)
