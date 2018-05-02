import math
import numpy as np

def rul_health_score(y_true, y_pred):
    s=0
    for i in range(len(y_true)):
        d = y_pred[i] - y_true[i]
        if d < 0:
            s+=math.e**(-d/13)-1
        else:
            s+=math.e**(d/10)-1
    return np.asscalar(s)

score_dict = {'rhs':lambda x,y : rul_health_score(x,y)}

def compute_score(score_name, y_true, y_pred):
    
    score = score_dict[score_name](y_true, y_pred)
    return score