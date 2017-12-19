
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

reviews.shape


# In[44]:

reviews.head()


# In[45]:

# Check for any nulls values
reviews.isnull().sum()


# In[48]:

reviews = reviews.drop(['id'],1)


# In[49]:

reviews.head()
reviews


# In[50]:

# Inspecting some of the reviews
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


# In[51]:


contractions = {
 	u'吻腚':u'稳定',
 	u'弓虽':u'强',
 	u'女干':u'奸',
 	u'示土':u'社',
 	u'禾口':u'和',
 	u'言皆':u'谐',
 	u'释永性':u'释永信',
 	u'大菊观':u'大局观',
 	u'yl':u'一楼',
 	u'cnm':u'草泥马',
 	u'CCTV':u'中央电视台',
 	u'CCAV':u'中央电视台',
 	u'ccav':u'中央电视台',
 	u'cctv':u'中央电视台',
 	u'qq':u'腾讯聊天账号',
 	u'QQ':u'腾讯聊天账号',
 	u'cctv':u'中央电视台',
 	u'CEO':u'首席执行官',
 	u'克宫':u'克里姆林宫',
 	u'PM2.5':u'细颗粒物',
 	u'pm2.5':u'细颗粒物',
 	u'SDR':u'特别提款权',
 	u'装13':u'装逼',
 	u'213':u'二逼',
 	u'13亿':u'十三亿',
 	u'巭':u'功夫',
 	u'孬':u'不好',
 	u'嫑':u'不要',
 	u'夯':u'大力',
 	u'芘':u'操逼',
 	u'烎':u'开火',
 	u'菌堆':u'军队',
 	u'sb':u'傻逼',
 	u'SB':u'傻逼',
 	u'Sb':u'傻逼',
 	u'sB':u'傻逼',
 	u'is':u'伊斯兰国',
 	u'isis':u'伊斯兰国',
 	u'ISIS':u'伊斯兰国',
 	u'ko':u'打晕',
 	u'你M':u'你妹',
 	u'你m':u'你妹',
 	u'震精':u'震惊',
 	u'返工分子':u'反共',
 	u'黄皮鹅狗':u'黄皮肤俄罗斯狗腿',
 	u'苏祸姨':u'苏霍伊',
 	u'混球屎报':u'环球时报',
 	u'屎报':u'时报',
 	u'jb':u'鸡巴',
 	u'j巴':u'鸡巴',
 	u'j8':u'鸡巴',
 	u'J8':u'鸡巴',
 	u'JB':u'鸡巴',
 	u'瞎BB':u'瞎说',
 	u'nb':u'牛逼',
 	u'牛b':u'牛逼',
 	u'牛B':u'牛逼',
 	u'牛bi':u'牛逼',
 	u'牛掰':u'牛逼',
 	u'苏24':u'苏两四',
 	u'苏27':u'苏两七',
 	u'痰腐集团':u'贪腐集团',
 	u'痰腐':u'贪腐',
 	u'反hua':u'反华',
 	u'<br>':u' ',
 	u'屋猫':u'五毛',
 	u'5毛':u'五毛',
 	u'傻大姆':u'萨达姆',
 	u'霉狗':u'美狗',
 	u'TMD':u'他妈的',
 	u'tmd':u'他妈的',
 	u'japan':u'日本',
 	u'P民':u'屁民',
 	u'八离开烩':u'巴黎开会',
 	u'傻比':u'傻逼',
 	u'潶鬼':u'黑鬼',
 	u'cao':u'操',
 	u'爱龟':u'爱国',
 	u'天草':u'天朝',
 	u'灰机':u'飞机',
 	u'张将军':u'张召忠',
 	u'大裤衩':u'中央电视台总部大楼',
 	u'枪毕':u'枪毙',
 	u'环球屎报':u'环球时报',
 	u'环球屎包':u'环球时报',
 	u'混球报':u'环球时报',
 	u'还球时报':u'环球时报',
 	u'人X日报':u'人民日报',
 	u'人x日报':u'人民日报',
 	u'清只县':u'清知县',
 	u'PM值':u'颗粒物值',
 	u'TM':u'他妈',
 	u'首毒':u'首都',
 	u'gdp':u'国内生产总值',
 	u'GDP':u'国内生产总值',
 	u'鸡的屁':u'国内生产总值',
 	u'999':u'红十字会',
 	u'霉里贱':u'美利坚',
 	u'毛子':u'俄罗斯人',
 	u'ZF':u'政府',
 	u'zf':u'政府',
 	u'蒸腐':u'政府',
 	u'霉国':u'美国',
 	u'狗熊':u'俄罗斯',
 	u'恶罗斯':u'俄罗斯',
 	u'我x':u'我操',
 	u'x你妈':u'操你妈',
 	u'p用':u'屁用',
 	u'胎毒':u'台独',
 	u'DT':u'蛋疼',
 	u'dt':u'蛋疼',
 	u'IT':u'信息技术',
 	u'1楼':u'一楼',
 	u'2楼':u'二楼',
 	u'2逼':u'二逼',
 	u'二b':u'二逼',
 	u'二B':u'二逼',
 	u'晚9':u'晚九',
 	u'朝5':u'朝五',
 	u'黄易':u'黄色网易',
 	u'艹':u'操',
 	u'滚下抬':u'滚下台',
 	u'灵道':u'领导',
 	u'煳':u'糊',
 	u'跟贴被火星网友带走啦':u'',
 	u'棺猿':u'官员',
 	u'贯猿':u'官员',
 	u'巢县':u'朝鲜',
 	u'死大林':u'斯大林',
 	u'无毛们':u'五毛们',
 	u'天巢':u'天朝',
 	u'普特勒':u'普京',
 	u'依拉克':u'伊拉克',
 	u'歼20':u'歼二零',
 	u'歼10':u'歼十',
 	u'歼8':u'歼八',
 	u'f22':u'猛禽',
 	u'p民':u'屁民',
 	u'钟殃':u'中央',
    u'３':u'三',
    "１": "一",
    u'２':u'二',
    u'４':u'四',
    u'５':u'五',
    u'６':u'六',
    u'７':u'七',
    u'８':u'八',
    u'９':u'九',
    u'０':u'零'
 }


# In[11]:

# 去除停用词  暂时没写
# text =u'听说你超级喜欢万众掘金小游戏啊啊啊'
# default_mode = jieba.cut(text,cut_all=False)
# stopw = [line.strip().decode('utf-8') for line in open('D:\\Python27\\stopword.txt').readlines()]
# print u'搜索引擎模式:',u'/'.join(set(default_mode)-set(stopw))


# In[57]:

def clean_text(text, remove_stopwords = True):
    #jieba.load_userdict("dict.txt")
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
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', '', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', text)
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\”', '',text)
    text = re.sub(r'\＃', '', text)
    text = re.sub(r'\“', '',text)
    text = re.sub(r'\《', '',text)
    text = re.sub(r'\》', '',text)
    
    
    text = jieba.cut(text)
    text = " ".join(text)
    
    return text


# In[58]:

clean_sum  = []
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
#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold:
        vocab_to_int[word] = value
        value +=1
    else:
        try:
            embeddings_index[word]
            vocab_to_int[word] = value
            value +=1
        except:
            continue


# In[70]:

value


# In[71]:

len(vocab_to_int)


# 阈值设置为10，不在词向量中的且出现超过10次，那咱们就得自己做它的映射向量了

# In[72]:

# Limit the vocab that we will use to words that appear ≥ threshold 
#dictionary to convert words to integers
# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


# In[53]:

for word, i in vocab_to_int.items():
    print(word)


# ### 定义word_embedding_matrix

# In[77]:


# # Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 400
nb_words = len(vocab_to_int)   # nb_words = 874

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)

for word, i in vocab_to_int.items():
    try:
        embeddings_index[word]
        word_embedding_matrix[i] = embeddings_index[word].dtype = 'float32'
    except:
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        #embeddings_index[word] = new_embedding
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

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


# In[80]:

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])  # DataFrame 表头（列的名称），表的内容（二维矩阵），索引（每行一个唯一的标记）。


# In[82]:

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())


# In[83]:

# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))


# In[84]:

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))


# In[85]:

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# sorted

# In[86]:

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 1000
max_summary_length = 20
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length): 
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])
        
# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


# In[107]:

print(1)


# ## Building the Model
# 
# <img src="f1.png" alt="FAO" width="490">
# 
# Bidirectional RNNs(双向网络)的改进之处便是，假设当前的输出(第t步的输出)不仅仅与前面的序列有关，并且还与后面的序列有关。
# 
# 例如：预测一个语句中缺失的词语那么就需要根据上下文来进行预测。Bidirectional RNNs是一个相对较简单的RNNs，是由两个RNNs上下叠加在一起组成的。输出由这两个RNNs的隐藏层的状态决定的

# ## Building the Model

# In[34]:

def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


# In[35]:

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[36]:

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state


# In[37]:

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_summary_length):
    '''Create the training logits'''
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer) 

    training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_summary_length)
    return training_logits


# In[38]:

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    '''Create the inference logits'''
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
                
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
                
    inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)
    
    return inference_logits


# In[39]:

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          rnn_size)
            
    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
                                                                    _zero_state_tensors(rnn_size, 
                                                                                        batch_size, 
                                                                                        tf.float32)) 
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  summary_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[40]:

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''
    
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        text_length, 
                                                        summary_length, 
                                                        max_summary_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers)
    
    return training_logits, inference_logits


# In[41]:

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[42]:

def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
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
epochs =20
batch_size = 64
rnn_size = 64
num_layers = 4
learning_rate = 0.001
keep_probability = 0.75


# In[44]:

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    
    # Load the model inputs    
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size)
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")


# ## 训练网络

# In[45]:

# Subset the data for training
start = 20000
end = start + 5000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:",len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[1]))


# In[47]:

print('train')
# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0001
display_step = 20 # Check training loss after every 20 batches
stop_early = 0 
stop = 5 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

update_loss = 0 
batch_loss = 0
summary_update_loss = [] # Record the update losses for saving improvements in the model

checkpoint = "Model/best_model.ckpt" 
print(1)
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    print(2)
    # If we want to continue training a previous session
    #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
    #loader.restore(sess, checkpoint)
    
    for epoch_i in range(1, epochs+1):
        
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
           
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(sorted_texts_short) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)
                
                # If the update loss is at a new minimum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!') 
                    stop_early = 0
                    saver = tf.train.Saver() 
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
            
                    
        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training.")
            break


# ## 测试效果

# In[42]:

def text_to_seq(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# In[1]:

# Create your own review or use one from the dataset
#input_sentence = "The coffee tasted great and was at such a good price! I highly recommend this to everyone!"
#text = text_to_seq(input_sentence)


random = np.random.randint(0,len(clean_texts))
input_sentence = clean_texts[random]
input_summary = clean_sum[random]
text = text_to_seq(clean_texts[random])

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
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      summary_length: [np.random.randint(5,20)], 
                                      text_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})[0] 

# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"] 

print('Original Text:', input_sentence)
print('Original Sum: ', input_summary)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))
print('  Input Sums:  {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))



