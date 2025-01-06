from datetime import datetime, timedelta
import numpy as np
import re

def is_empty(data):
    if (data is None) or (not data) or (data is np.nan) or (data == '-') or (data == '--'):
        return True
    else:
        return False

def is_all_alpha(s):
    return bool(re.fullmatch(r'[A-Za-z]+', s))

def date_add_time(time1, time2):
    try:
        if is_empty(time1):
            return '--'
        dt1 = str2date(time1)
        if is_empty(time2):
            td2 = timedelta(hours=0, minutes=0, seconds=0)
        else:
            tmp = time2.count(':')
            hours, minutes, seconds = 0, 0, 0
            if tmp == 2:
                hours, minutes, seconds = map(int, time2.split(':'))
            elif tmp == 1:
                hours, minutes = map(int, time2.split(':'))
                seconds = 0
            td2 = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        result = dt1 + td2
        return result.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print('Error in function date_add_time')
        print(e)
        return '--'

def datetime_add_minute(time1,minutes):
    try:
        dt1 = str2datetime(time1)
        td2 = timedelta(minutes=minutes)
        result = dt1 + td2
        return result.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print('Error in function datetime_add_minute')
        print(e)
        return '--'

def datetime2date(time1):
    try:
        if is_empty(time1):
            return '--'
        dt1 = str2date(time1)
        return dt1.strftime("%Y-%m-%d")
    except Exception as e:
        print('Error in function datetime2date')
        print(e)
        return '--'

def days_between(time1,time2):
    try:
        if is_empty(time1) or is_empty(time2):
            return '--'
        dt1 = str2date(time1)
        dt2 = str2date(time2)
        return (dt2-dt1).days
    except Exception as e:
        print('Error in function days_between')
        print(e)
        return '--'

def hours_between(time1, time2):
    try:
        if is_empty(time1) or is_empty(time2):
            return '--'
        dt1 = str2datetime(time1)
        dt2 = str2datetime(time2)
        return np.ceil((dt2-dt1).seconds/3600)+(dt2-dt1).days*24
    except Exception as e:
        print('Error in function hours_between')
        print(e)
        return '--'

def str2date(time):
    if is_empty(time):
        return None
    if ' ' in time:
        time = time.split(' ')[0]
    if '-' in time:
        date_format = "%Y-%m-%d"
    elif '/' in time:
        date_format = "%Y/%m/%d"
    else:
        return None
    return datetime.strptime(time, date_format)

def str2date_mdy(time):
    if is_empty(time):
        return None
    if ' ' in time:
        time = time.split(' ')[0]
    if '/' in time:
        date_format = "%m/%d/%Y"
    else:
        return None
    return datetime.strptime(time,date_format)

def mdy2ymd(time):
    # 针对一些数据
    if ' ' in time:
        time = time.split(' ')[0]
    if '/' in time:
        dt1 = str2date_mdy(time)
        if dt1 is not None:
            return dt1.strftime("%Y-%m-%d")
        else:
            return ''
    else:
        return time

def str2datetime(time):
    date_format = "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(time, date_format)

def row_in_excel_to_string(row):
    string = ''
    for cell in row:
        if cell:
            string += str(cell)
    return string

def toFloat(f):
    try:
        f = float(f)
        return f
    except Exception:
        return f