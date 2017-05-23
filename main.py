import matplotlib.pyplot as plt

from params import *

# HIT = read_data()

# print('\nTensor u_iu_j')
# for i in ['u', 'v', 'w']:
#     for j in ['u', 'v', 'w']:
#         HIT[i+j] = np.multiply(HIT[i], HIT[j])

# LES = filter3d(data=HIT, scale_k=10, filename='LES')
# TEST = filter3d(data=HIT, scale_k=5, filename='TEST')

# # Load data
LES = np.load('./data/LES.npz')
TEST = np.load('./data/TEST.npz')

# map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
# plot.imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'fourier_tophat')

# # Gradients
# print("\nGradients")
# A_TEST = dict()
# A_TEST['uu'], A_TEST['uv'], A_TEST['uw'] = np.gradient(TEST['u'], dx[0], dx[1], dx[2])
# A_TEST['vu'], A_TEST['vv'], A_TEST['vw'] = np.gradient(TEST['v'], dx[0], dx[1], dx[2])
# A_TEST['wu'], A_TEST['wv'], A_TEST['ww'] = np.gradient(TEST['w'], dx[0], dx[1], dx[2])
#
# A_LES = dict()
# A_LES['uu'], A_LES['uv'], A_LES['uw'] = np.gradient(LES['u'], dx[0], dx[1], dx[2])
# A_LES['vu'], A_LES['vv'], A_LES['vw'] = np.gradient(LES['v'], dx[0], dx[1], dx[2])
# A_LES['wu'], A_LES['wv'], A_LES['ww'] = np.gradient(LES['w'], dx[0], dx[1], dx[2])
#
# # Strain tensor
# print("\nStrain tensor TEST")
# S_TEST = dict()
# for i in ['u', 'v', 'w']:
#     for j in ['u', 'v', 'w']:
#         S_TEST[i+j] = 0.5 * (A_TEST[i+j] + A_TEST[j+i])
#
# print("\nTau TEST")
# tau_TEST = dict()
# for i in ['u', 'v', 'w']:
#     for j in ['u', 'v', 'w']:
#         tau_TEST[i+j] = TEST[i+j] - np.multiply(TEST[i], TEST[j])
#
# print('\nT_ijS_ij')
# TS = 0
# for i in ['u', 'v', 'w']:
#     for j in ['u', 'v', 'w']:
#         TS += np.multiply(S_TEST[i+j], tau_TEST[i+j])
# #
# print('\nWriting files')
# file_T = './data/T.npz'
# file_TS = './data/TS.npz'
# #
# np.savez(file_T, uu=tau_TEST['uu'], uv=tau_TEST['uv'], uw=tau_TEST['uw'])
# np.savez(file_TS, TS=TS)

tau_TEST = np.load('./data/T.npz')
TS = np.load('./data/TS.npz')['TS']
print('\nPlotting')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 6))
titles = [r'$T_{11}$', r'$T_{12}$', r'$T_{13}$']
ax1.hist(tau_TEST['uu'].flatten(), bins=100, normed=1, alpha=0.4)
ax2.hist(tau_TEST['uv'].flatten(), bins=100, normed=1, alpha=0.4)
ax3.hist(tau_TEST['uw'].flatten(), bins=100, normed=1, alpha=0.4)

for ind, ax in enumerate([ax1, ax2, ax3]):
    ax.set_xlabel(titles[ind])
ax1.axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
ax1.set_ylabel('pdf')
ax3.set_yscale('log', nonposy='clip')
fig.tight_layout()
# plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.show()


# plt.figure(figsize=(8, 8))
# plt.hist(tau_TEST['uu'].flatten(), bins=100, normed=1,  alpha=0.4)
# plt.yscale('log', nonposy='clip')
# plt.xlabel(r'$T_{11}$')
# plt.ylabel(r'pdf($T_{11}$)')
# plt.axis(xmin=-1.1, xmax=1.1, ymin=1e-5)
# plt.show()
#
# plt.figure(figsize=(8, 8))
# plt.hist(TS.flatten(), bins=500, normed=1,  alpha=0.4)
# plt.yscale('log', nonposy='clip')
# plt.xlabel(r'$T_{ij}S^T_{ij}$')
# plt.ylabel(r'pdf($T_{ij}S^T_{ij}$)')
# plt.axis(xmin=-5, xmax=4, ymin=1e-5)
# plt.show()
