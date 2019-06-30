# text-sentiment-analysis
text sentiment analysis
针对文本情感分析，借鉴TextCNN的部分思想，提出了一种新的深度学习模型——Deep-TextCNN,与TextCNN不同，本模型将每个文本排列成一行，输入的格式为：
[batch_size, height, width, channels]，其中height = 1，width为文本词语个数，channels为词向量维度。模型分为三个部分，第一部分是：分别使用
1*1，1*2,1*3,1*5,1*7的卷积核对输入进行卷积操作，并按照最后一个维度进行拼接。第二部分使用了4*（2个CNN+1个膨胀卷积），最后一部分为两层的FC。
