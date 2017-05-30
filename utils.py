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

