from params import *
import random as rand
import calculate
import plot

def ABC(tau_exact, test_field, N):
    C_s = [rand.random() for i in range(N)]
    for i in C_s:
        tau = calculate.Reynolds_stresses_from_Cs(test_field, i)
        print(tau['uu'].shape, tau['uv'].shape, tau['uw'].shape)
        plot.T_TEST(tau)