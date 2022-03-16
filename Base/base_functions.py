
import numpy as np
from scipy import stats


def p_values_null_coef(coefficients):
    #inverse of quantile
    return stats.percentileofscore(coefficients,0)

def p_values_arg_coef(coefficients,arg):
    #inverse of quantile
    return stats.percentileofscore(coefficients,arg)
