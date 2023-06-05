import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.preprocessing.text import Tokenizer
import faiss
import numpy as np


# 学習用データベクトル化
def toVector(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer.texts_to_matrix(data, "tfidf")


# 学習用兼、投入データ。AOAIの場合、以下のようにスペースで区切らずとも AOAI 側の Tokenizer でよしなに 1536 次元に分割してくれる
target_texts = [
    "好きな 食べ物は 何ですか?",
    "どこに お住まい ですか?",
    "朝の 電車は 混みますね",
    "今日は 良い お天気 ですね",
    "最近 景気 悪い ですね",
    "今日は 雨振らなくて よかった",
]

# ベクトル化
embededs = np.array(toVector(target_texts)).astype("float32")
target = embededs[:5]
input = embededs[5:]

# 近似検索
index = faiss.IndexFlatL2(len(target[0]))
index.add(target)
_, result = index.search(input, 3)  # 上位3件を取得

# プロンプト生成イメージ
print(
    "以下のコンテキストを用いて回答して下さい\n\nコンテキスト:" + target_texts[result[0][0]] + "\n\n{ここに質問文が来る}"
)
