
from konlpy.tag import Okt

okt =Okt()
token = okt.morphs("나는 자연어 처리를 배운다")
print(token)


# 각 토큰(나눠진 단어)에 대해서 고유 인덱스 부여
word2index = {}
for voca in token :
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
print(word2index)


def one_hot_encoding(word, word2index) :
    one_hot_vector = [0]*(len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

one_hot_encoding("자연어",word2index)


# keras 이용한 one hot encoding

# 각 단어에 대한 인덱스 출력
text = "나랑 별보러 가지 않을래 어디든 좋으니 나와 가줄래"

from keras_preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index)

# 위 과정은 단어집합에 있는 단어들의 텍스트만 있다면, texts_to_sequences()로
# 바로 인덱스 나열이 가능

text2 = "나랑 별보러 가지 않을래 어디든 좋으니 나와 가줄래"
x = t.texts_to_sequences([text2])
print(x)