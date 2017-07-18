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

def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf

def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    return pdf



