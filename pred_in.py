import numpy as np
import calendar as cl
from datetime import datetime, date

def prediction_input():
    day_dict = {}
    prev = 365 
    next = 365
    for i in range(1,13):
        next += cl.monthrange(2022, i)[1]
        day_dict[cl.month_name[i].lower()] = [[x, i, date(2022, i ,x-prev+1).timetuple().tm_wday] for x in range(prev,next)]
        prev = next
    return(day_dict)