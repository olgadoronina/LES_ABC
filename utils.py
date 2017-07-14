from params import *


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:05.2f}".format(int(minutes), seconds), '\t', label)


def read_data():
    data = dict()
    data['u'] = np.reshape(np.fromfile(datafile_u, dtype=np.float32), tuple(N_points))
    data['v'] = np.reshape(np.fromfile(datafile_v, dtype=np.float32), tuple(N_points))
    data['w'] = np.reshape(np.fromfile(datafile_w, dtype=np.float32), tuple(N_points))

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

def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf

def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    return pdf

def form_C_array(n):
    C_array = []
    for i in range(N):
        C = []
        for j in range(n):
            C.append(rand.uniform(C_limits[j][0], C_limits[j][1]))
        C_array.append(C)
    return C_array

