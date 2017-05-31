from params import *
import utils
import filter
import plot
import calculate
import abc_function
import random as rand
# HIT = utils.read_data()
# print('\nTensor u_iu_j')
# for i in ['u', 'v', 'w']:
#     for j in ['u', 'v', 'w']:
#         HIT[i+j] = np.multiply(HIT[i], HIT[j])
#
# print('\nFilter data')
# LES = filter.filter3d(data=HIT, scale_k=LES_scale)
# TEST = filter.filter3d(data=HIT, scale_k=TEST_scale)

print("\nLoad data")
LES = np.load('./data/LES.npz')
TEST = np.load('./data/TEST.npz')
#
# # map_bounds = np.linspace(np.min(HIT['u'][:, :, 127]), np.max(HIT['u'][:, :, 127]), 20)
# # plot.imagesc([HIT['u'][:, :, 127], LES['u'][:, :, 127], TEST['u'][:, :, 127]], map_bounds, 'fourier_tophat')
#
# print("\nStrain tensors")
# S_TEST = calculate.strain_tensor(TEST)
# S_LES = calculate.strain_tensor(LES)
#
# print("\nT TEST")
# # T_TEST = np.load('./data/T.npz')
# T_TEST = calculate.Reynolds_stresses(TEST)
T_LES = calculate.Reynolds_stresses_from_DNS(LES)
# # plot.T_TEST(T_TEST)
#
# print('\nT_ijS_ij')
# # TS = np.load('./data/TS.npz')['TS']
# # TS = calculate.scalar_product(T_TEST, S_TEST)
# # plot.TS(TS)
#
# # print('\nWriting files')
# # np.savez('./data/T.npz', uu=T_TEST['uu'], uv=T_TEST['uv'], uw=T_TEST['uw'])
# # np.savez('./data/TS.npz' TS=TS)
#
# print('\nCalculate Smagorinsky constant')
# # C_s = calculate.Smagorinsky_constant_dynamic(LES, TEST, S_LES, S_TEST)
# # C_s = calculate.Smagorinsky_constant_from_DNS(LES, S_LES)
# # C_s = calculate.Smagorinsky_constant_from_DNS(TEST, S_TEST)

LES_sp = utils.sparse_dict(LES, 16)
TEST_sp = utils.sparse_dict(TEST, 16)
T_LES_sp = calculate.Reynolds_stresses_from_DNS(LES_sp)

# plot.tau_tau_sp(T_LES, T_LES_sp)
plot.tau_sp(T_LES_sp)
# abc_function.ABC(T_LES_sp, TEST_sp, 5)
