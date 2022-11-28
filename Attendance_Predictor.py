#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

def word_indexes(descriptions):
    word_set = set()
    for sen in descriptions:
        words = sen.lower().split(" ")
        for word in words:
            word_set.add(word)
    word_to_index = {}
    count = 0
    for word in word_set:
        word_to_index[word] = count
        count += 1
    return word_to_index

def tf_idf_mat(descriptions, word_to_index):

    doc_term_matrix = []
    for sen in descriptions:
        counts = [0] * len(word_to_index)
        for word in sen.lower().split(" "):
            if word in word_to_index:
                counts[word_to_index[word]] += 1
        doc_term_matrix.append(counts)

    doc_term_matrix = np.array(doc_term_matrix)

    tf_matrix = doc_term_matrix / np.sum(doc_term_matrix, axis = 1, keepdims = True)
    idf_matrix = np.log(doc_term_matrix.shape[0] / (np.sum(doc_term_matrix > 0, axis = 0, keepdims = True)))
    return tf_matrix * idf_matrix

def predict(desc, tf_idf_matrix, word_to_index):
    d_vector = [0] * len(word_to_index)
    for word in desc.lower().split(" "):
        if word in word_to_index:
            d_vector[word_to_index[word]] += 1
    d_vector = np.array(d_vector)
    max_cos = -1
    max_ind_cos = -1
    count = 0
    for doc in tf_idf_matrix:
        cos = np.dot(doc, d_vector) / np.sqrt(np.sum(doc**2))
        if cos > max_cos:
            max_cos = cos
            max_ind_cos = count
        count += 1
    return max_ind_cos


df = pd.read_csv("Party_data_updated.csv")
p = df[df["attendeesCount"] > 0]["description"]
turnout = np.where(df[df["attendeesCount"] > 0]["attendeesCount"] > 4, "big", "small")
ptrain = p.iloc[:82]
ptest = p.iloc[82:]
turnout_train = turnout[:82]
turnout_test = turnout[82:]
words = word_indexes(ptrain)
mat = tf_idf_mat(ptrain, words)
#rint(df[df['attendeesCount'].notna()]["description"].describe(include = "all"))
#print(df["description"].describe(include = "all"))
#print(df[df["attendeesCount"] > 0].describe())
confusion = pd.DataFrame({"true_big":{"predicted_small":0, "predicted_big":0}, "true_small":{"predicted_big":0, "predicted_small":0}})
for doc,val in zip(ptest, turnout_test):
    prediction = predict(doc, mat, words)
    if turnout_train[prediction] == "big" and val == "big":
        confusion.at["predicted_big", "true_big"] += 1
    if turnout_train[prediction] == "big" and val == "small":
        confusion.at["predicted_big", "true_small"] += 1
    if turnout_train[prediction] == "small" and val == "big":
        confusion.at["predicted_small", "true_big"] += 1
    if turnout_train[prediction] == "small" and val == "small":
        confusion.at["predicted_small", "true_small"] += 1
conf_names = [["PPTP", "PNTN"], ["PPTN", "PNTP"]]
sns.set_theme(style='dark')
sns.heatmap(confusion, annot = True, cmap = sns.color_palette("blend:#00ADFF,#D705F2", as_cmap=True))
correct = confusion.at["predicted_big", "true_big"] + confusion.at["predicted_small", "true_small"]
incorrect = confusion.at["predicted_small", "true_big"] + confusion.at["predicted_big", "true_small"]
print(f"Accuracy: {correct / (correct + incorrect)}")

#print(words)


#%%


doc_term_matrix = []
for sen in ptrain:
    counts = [0] * len(words)
    for word in sen.lower().split(" "):
        if word in words:
            counts[words[word]] += 1
    doc_term_matrix.append(counts)

doc_term_matrix = np.array(doc_term_matrix)
tf_matrix = doc_term_matrix / np.sum(doc_term_matrix, axis = 1, keepdims = True)
idf_matrix = np.log(doc_term_matrix.shape[0] / (np.sum(doc_term_matrix > 0, axis = 0, keepdims = True)))
tf_idf_matrix = tf_matrix * idf_matrix
x = list(words)
y = list()
for word in x:
    y.append(idf_matrix[0, words[word]])
x = [re.sub("\W+", " ", a).strip() for a in x]

x = [a if a else "emoji" for a in x]
x,y = zip(*sorted(zip(x,y), key=lambda a: -a[1]))
#plt.figure()
#plt.barh(x[:20], y[:20])
#plt.ylabel("Words")
#plt.xlabel("Inverse Document Frequency") #log of inverse of number of documents term appears in over total number of documents
#plt.title("Most Significant Words")
#plt.show()
print(f"{x[:20]} {y[:20:]}")
# %%
