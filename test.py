# coding: utf-8

# # Summarizing Text with Amazon Reviews

# 数据集：Amazon 500000评论
#
#
# 本节内容:
# - 数据预处理
# - 构建Seq2Seq模型
# - 训练网络
# - 测试效果
#
# seq2seq教程: https://github.com/j-min/tf_tutorial_plus/tree/master/RNN_seq2seq/contrib_seq2seq

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
import jieba
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

print('TensorFlow Version: {}'.format(tf.__version__))

# In[2]:

import csv
from gensim.models import Word2Vec
import gensim

# ## Insepcting the Data

# In[42]:

reviews = pd.read_csv("data_full.csv")

# In[43]:

reviews = reviews.drop(['id'], 1)


contractions = {
    u'吻腚': u'稳定',
    u'弓虽': u'强',
    u'女干': u'奸',
    u'示土': u'社',
    u'禾口': u'和',
    u'言皆': u'谐',
    u'释永性': u'释永信',
    u'大菊观': u'大局观',
    u'yl': u'一楼',
    u'cnm': u'草泥马',
    u'CCTV': u'中央电视台',
    u'CCAV': u'中央电视台',
    u'ccav': u'中央电视台',
    u'cctv': u'中央电视台',
    u'qq': u'腾讯聊天账号',
    u'QQ': u'腾讯聊天账号',
    u'cctv': u'中央电视台',
    u'CEO': u'首席执行官',
    u'克宫': u'克里姆林宫',
    u'PM2.5': u'细颗粒物',
    u'pm2.5': u'细颗粒物',
    u'SDR': u'特别提款权',
    u'装13': u'装逼',
    u'213': u'二逼',
    u'13亿': u'十三亿',
    u'巭': u'功夫',
    u'孬': u'不好',
    u'嫑': u'不要',
    u'夯': u'大力',
    u'芘': u'操逼',
    u'烎': u'开火',
    u'菌堆': u'军队',
    u'sb': u'傻逼',
    u'SB': u'傻逼',
    u'Sb': u'傻逼',
    u'sB': u'傻逼',
    u'is': u'伊斯兰国',
    u'isis': u'伊斯兰国',
    u'ISIS': u'伊斯兰国',
    u'ko': u'打晕',
    u'你M': u'你妹',
    u'你m': u'你妹',
    u'震精': u'震惊',
    u'返工分子': u'反共',
    u'黄皮鹅狗': u'黄皮肤俄罗斯狗腿',
    u'苏祸姨': u'苏霍伊',
    u'混球屎报': u'环球时报',
    u'屎报': u'时报',
    u'jb': u'鸡巴',
    u'j巴': u'鸡巴',
    u'j8': u'鸡巴',
    u'J8': u'鸡巴',
    u'JB': u'鸡巴',
    u'瞎BB': u'瞎说',
    u'nb': u'牛逼',
    u'牛b': u'牛逼',
    u'牛B': u'牛逼',
    u'牛bi': u'牛逼',
    u'牛掰': u'牛逼',
    u'苏24': u'苏两四',
    u'苏27': u'苏两七',
    u'痰腐集团': u'贪腐集团',
    u'痰腐': u'贪腐',
    u'反hua': u'反华',
    u'<br>': u' ',
    u'屋猫': u'五毛',
    u'5毛': u'五毛',
    u'傻大姆': u'萨达姆',
    u'霉狗': u'美狗',
    u'TMD': u'他妈的',
    u'tmd': u'他妈的',
    u'japan': u'日本',
    u'P民': u'屁民',
    u'八离开烩': u'巴黎开会',
    u'傻比': u'傻逼',
    u'潶鬼': u'黑鬼',
    u'cao': u'操',
    u'爱龟': u'爱国',
    u'天草': u'天朝',
    u'灰机': u'飞机',
    u'张将军': u'张召忠',
    u'大裤衩': u'中央电视台总部大楼',
    u'枪毕': u'枪毙',
    u'环球屎报': u'环球时报',
    u'环球屎包': u'环球时报',
    u'混球报': u'环球时报',
    u'还球时报': u'环球时报',
    u'人X日报': u'人民日报',
    u'人x日报': u'人民日报',
    u'清只县': u'清知县',
    u'PM值': u'颗粒物值',
    u'TM': u'他妈',
    u'首毒': u'首都',
    u'gdp': u'国内生产总值',
    u'GDP': u'国内生产总值',
    u'鸡的屁': u'国内生产总值',
    u'999': u'红十字会',
    u'霉里贱': u'美利坚',
    u'毛子': u'俄罗斯人',
    u'ZF': u'政府',
    u'zf': u'政府',
    u'蒸腐': u'政府',
    u'霉国': u'美国',
    u'狗熊': u'俄罗斯',
    u'恶罗斯': u'俄罗斯',
    u'我x': u'我操',
    u'x你妈': u'操你妈',
    u'p用': u'屁用',
    u'胎毒': u'台独',
    u'DT': u'蛋疼',
    u'dt': u'蛋疼',
    u'IT': u'信息技术',
    u'1楼': u'一楼',
    u'2楼': u'二楼',
    u'2逼': u'二逼',
    u'二b': u'二逼',
    u'二B': u'二逼',
    u'晚9': u'晚九',
    u'朝5': u'朝五',
    u'黄易': u'黄色网易',
    u'艹': u'操',
    u'滚下抬': u'滚下台',
    u'灵道': u'领导',
    u'煳': u'糊',
    u'跟贴被火星网友带走啦': u'',
    u'棺猿': u'官员',
    u'贯猿': u'官员',
    u'巢县': u'朝鲜',
    u'死大林': u'斯大林',
    u'无毛们': u'五毛们',
    u'天巢': u'天朝',
    u'普特勒': u'普京',
    u'依拉克': u'伊拉克',
    u'歼20': u'歼二零',
    u'歼10': u'歼十',
    u'歼8': u'歼八',
    u'f22': u'猛禽',
    u'p民': u'屁民',
    u'钟殃': u'中央',
    u'３': u'三',
    "１": "一",
    u'２': u'二',
    u'４': u'四',
    u'５': u'五',
    u'６': u'六',
    u'７': u'七',
    u'８': u'八',
    u'９': u'九',
    u'０': u'零'
}


# In[11]:

# 去除停用词  暂时没写
# text =u'听说你超级喜欢万众掘金小游戏啊啊啊'
# default_mode = jieba.cut(text,cut_all=False)
# stopw = [line.strip().decode('utf-8') for line in open('D:\\Python27\\stopword.txt').readlines()]
# print u'搜索引擎模式:',u'/'.join(set(default_mode)-set(stopw))


# In[57]:

def clean_text(text, remove_stopwords=True):
    # jieba.load_userdict("dict.txt")
    text = jieba.cut(text)
    text = " ".join(text)

    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    text = re.sub('（.*?）', '', text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', '', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\”', '', text)
    text = re.sub(r'\＃', '', text)
    text = re.sub(r'\“', '', text)
    text = re.sub(r'\《', '', text)
    text = re.sub(r'\》', '', text)

    text = jieba.cut(text)
    text = " ".join(text)

    return text


# In[58]:

clean_sum = []
for summary in reviews.Summary:
    clean_sum.append(clean_text(summary))

print('summary ok')

# In[60]:

clean_sum[2]

# In[61]:

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

# In[63]:

clean_texts[2]


# We will remove the stopwords from the texts because they do not provide much use for training our model. However, we will keep them for our summaries so that they sound more like natural phrases.

# In[64]:

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# In[65]:

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_sum)
count_words(word_counts, clean_texts)

print("Size of Vocabulary:", len(word_counts))

# ## 使用构建好的词向量

# embeddings_index 是一个model

# In[68]:

embeddings_index = Word2Vec.load('wordembedding/wiki.zh.text.model')

# In[69]:

threshold = 10
# dictionary to convert words to integers
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold:
        vocab_to_int[word] = value
        value += 1
    else:
        try:
            embeddings_index[word]
            vocab_to_int[word] = value
            value += 1
        except:
            continue

# In[70]:

# In[71]:


# 阈值设置为10，不在词向量中的且出现超过10次，那咱们就得自己做它的映射向量了

# In[72]:

# Limit the vocab that we will use to words that appear ≥ threshold
# dictionary to convert words to integers
# Special tokens that will be added to our vocab
codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts), 4) * 100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))

# In[53]:

# ### 定义word_embedding_matrix

# In[77]:


# # Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 400
nb_words = len(vocab_to_int)  # nb_words = 874

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)

for word, i in vocab_to_int.items():
    try:
        embeddings_index[word]
        word_embedding_matrix[i] = embeddings_index[word].dtype = 'float32'
    except:
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        # embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

print(len(word_embedding_matrix))

# In[74]:

word_embedding_matrix


# In[78]:

def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# In[79]:

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_sum, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) * 100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


sorted_summaries = []
sorted_texts = []
max_text_length = 1000
max_summary_length = 20
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

# In[107]:

print(1)


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[42]:

def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# In[46]:

# Set the Hyperparameters
epochs = 20
batch_size = 64
rnn_size = 64
num_layers = 4
learning_rate = 0.001
keep_probability = 0.75

# In[44]:



# ## 测试效果

# In[42]:

def text_to_seq(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# In[1]:

# Create your own review or use one from the dataset
input_sentence = "马丽竹小朋友只有十八岁，爱喝咖啡还睡不着觉，还得在家看孩子"
text = text_to_seq(input_sentence)


random = np.random.randint(0, len(clean_texts))
#input_sentence = clean_texts[random]
input_summary = clean_sum[random]
#text = text_to_seq(clean_texts[random])

checkpoint = "./Model/best_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    # Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      summary_length: [np.random.randint(5, 20)],
                                      text_length: [len(text)] * batch_size,
                                      keep_prob: 1.0})[0]

# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"]

print('Original Text:', input_sentence)
#print('Original Sum: ', input_summary)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))


# Examples of reviews and summaries:
# - Review(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
# - Summary(1): great coffee
#
#
# - Review(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!
# - Summary(2): omg gross gross
#
#
# - Review(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
# - Summary(3): love it

# In[ ]:



