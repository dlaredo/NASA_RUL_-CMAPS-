import math
import numpy as np

def rul_health_score(y_true, y_pred):
    s=0

    d = y_pred - y_true

    #print(d)

    for i in range(len(d)):
        if d[i] < 0:
            s+=math.e**(-d[i]/13)-1
        else:
            s+=math.e**(d[i]/10)-1

    s = s/len(d)

    return s

score_dict = {'rhs':lambda x,y : rul_health_score(x,y)}

def compute_score(score_name, y_true, y_pred):
    
    score = score_dict[score_name](y_true, y_pred)
    return score