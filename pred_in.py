import numpy as np
import calendar as cl
from datetime import datetime, date

#Constructing X variables for all the days in the year 2022
def prediction_input():
    day_dict = {}
    prev = 365 
    next = 365
    for i in range(1,13):
        next += cl.monthrange(2022, i)[1]
        day_dict[cl.month_name[i].lower()] = [[x, i, date(2022, i ,x-prev+1).timetuple().tm_wday] for x in range(prev,next)]
        prev = next
    return(day_dict)

# for day_dict--> key: month name in lowercase; values: X values with 3 features for the {key} month