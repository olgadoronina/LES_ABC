from params import *


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:05.2f}".format(int(minutes), seconds), '\t', label)


def read_data():
    data = dict()
    data['u'] = np.reshape(np.fromfile('./data/HIT_u.bin', dtype=np.float32), tuple(Nx))
    data['v'] = np.reshape(np.fromfile('./data/HIT_v.bin', dtype=np.float32), tuple(Nx))
    data['w'] = np.reshape(np.fromfile('./data/HIT_w.bin', dtype=np.float32), tuple(Nx))

    # to put x index in first place
    for key, value in data.items():
        data[key] = np.swapaxes(value, 0, 2)
    return data


def sparse_array(data, M):
    if data.shape[0] % M:
        print('Error: utils.sparse_dict(): Nonzero remainder')
    n_th = int(data.shape[0] / M)
    sparse_data = data[::n_th, ::n_th, ::n_th].copy()
    return sparse_data


def sparse_dict(data, M):
    sparse = dict()
    for key, value in data.items():
        sparse[key] = sparse_array(value, M)
    return sparse

