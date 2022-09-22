"""基于词袋模型的贝叶斯过滤垃圾邮件"""
import re

# 定义一个邮件实例
mysent = 'This book is the best book on python or M.L. I have ever laid eyes upon'
mysent = mysent.lower()
# 使用正则表达式获得想要的词条分割
regex = re.compile(r'\W+')
listoftokens = regex.split(mysent)
print(f"The word list of eamil: \n{listoftokens}")

# 导入一个真的邮件进行处理
