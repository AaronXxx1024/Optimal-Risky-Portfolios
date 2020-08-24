

def isiterable(obj):
    """
    Test obj is iterable or not
    :param obj: any obj
    :return: boole (True or False)
    """
    try:
        iter(obj)
        return True
    except TypeError:  # not iterable
        return False

# import time
# start = time.process_time()
# content
# end = time.process_time()
# print('Running time: %s Seconds' % (end-start))

def check_wd():
    """
    :return:
    """
    import os
    wd = os.getcwd()
    return print(wd)

def change_wd(path:str):
    """
    改变当前working directory到指定的path
    :param path: path of designated working directory, should be str
    :return: ***
    """
    from os import chdir
    return chdir(path)

def file_in_path(path=None):
    """
    :param path: 字符串路径，默认为None
    :return: 默认为None,返回工作目录下文件，给予字符串路径后，返回指定路径下文件名
    """
    from os import listdir
    return print(listdir(path))

def stock_quote(name=None,
                source=None,
                start=None,
                end=None,
                percentage=True):
    """
    调用pandas_datareader的API以获取相应股票数据， 加入微小的修改以适应个人需要
    :param name: follow YahooDailyReader standard
    :param source: follow YahooDailyReader standard
    :param start: follow YahooDailyReader standard
    :param end: follow YahooDailyReader standard
    :param percentage: Boole, if 'True', return daily return, or return original daily price
    :return: 返回每日收益率或者daily price
    """
    from pandas_datareader.data import DataReader
    raw = DataReader(name=name, data_source=source, start=start, end=end)
    if percentage is True:
        data = raw.pct_change()
        data = data.iloc[1:, ]
        return data
    else:
        return raw

