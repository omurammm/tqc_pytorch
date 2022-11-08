def filename_to_tuple(name):
    b = int(name.split('bottom')[1][0])
    t = int(name.split('top')[1][0])
    m = int(name.split('mean')[1][0])
    return (b, t, m)

def result_dict(files):
    result = {}
    for file in files:
        params = filename_to_tuple(file)
        res = np.load(file)
        result[params] = res
    return result

def ma(y, n):
    b=np.ones(n)/n

    y2=np.convolve(y, b, mode='same')#移動平均
    return y2

def print_best(res, ma_n=1):
    minlen = 99999999
    maxlen = 0
    for v in res.values():
        minlen = len(v) if len(v) < minlen else minlen
        maxlen = len(v) if len(v) > maxlen else maxlen
    print('minimum len:', minlen, 'step:', minlen*1e3)
    print('minimum len:', maxlen, 'step:', maxlen*1e3)
    print('(b, t, m)')
    for k, v in sorted(res.items()):
        print(k, round(ma(v, ma_n)[:minlen].max(), 0))
        
def plot_keys(res, keys, ma_n=1):
    for k, v in sorted(res.items()):
        if k in keys:
            plt.plot(ma(v, ma_n), label=str(k))
    plt.legend()
    plt.show()