import time


def tprint(string, end='\n'):
    print(f'{time.strftime("%H:%M:%S", time.localtime(time.time()))}    {string}', end=end)
