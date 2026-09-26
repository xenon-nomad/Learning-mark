import math
def HR(rec_list,trust_set,k):
    Topk=set(rec_list[:k])
    return 1 if Topk&trust_set else 0
def Recall(rec_list,trust_set,k):
    Topk=set(rec_list[:k])
    hits=len(Topk&trust_set)
    return hits/len(trust_set)
def NDCG(rec_list,trust_set,k):
    dcg=0.0
    for rank,item in enumerate(rec_list[:k],start=1):
        if item in trust_set:
            dcg+=1/math.log2(rank+1)
    ideal_len=min(len(trust_set),k)
    idcg=0.0
    for r in range(1,ideal_len+1):
        idcg+=1/math.log2(r+1)

    return dcg/idcg
