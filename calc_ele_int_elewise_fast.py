import numpy as np
#import scipy.sparse
import time
#from tqdm import tqdm
from numba import jit, prange
# Import the parameters
from input import *

start_time_all = time.time()

NumDiff2modes=int((N)*(N-1)/2)
wavefn_save=np.load(calcdir + '/rho.npz')['rho']
index=np.load(calcdir + '/index.npy')

r = np.linspace(0,l,r_resolution)

# generate index for all basis
basis_index=np.zeros((int(4*N+2+2*NumDiff2modes),3))

def get_mode_func(alpha):
    mode_func = np.sqrt( (w[:,np.newaxis]) / (epsilon*l) ) \
                          * np.sin(np.pi*alpha[:,np.newaxis]*r/l)
    return mode_func

# generate index for alpha^1beta^1
index_2 = np.array([0, 0])
for i in range(N):
    for j in range(i + 1, N):
        index_2 = np.vstack([index_2, np.array([i, j])])

# Calculation part
mode_func = get_mode_func(alpha)  # has a shape of ( # of modes, r_res)

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_vacuum(wavefn_save, wavefn_conj):
    g0 = np.zeros(r.size) + 0j
    # Calculate |g,0>
    # to |g, 0>
    # g0=np.sum(wavefn_conj[t_index, 0]* mode_func**2 *wavefn_save[t_index, 0], axis=0)
    # to |g, alpha2>
    for i in prange(int(N + 1), int(2 * N) + 1):
        index = int(i - (N + 1))
        g0 += np.sqrt(2)*wavefn_conj[i]*mode_func[index]**2*wavefn_save[0]
    # to |g, alpha1beta1>
    if N != 1:
        for i in prange(2*int(N)+1, 2*int(N)+NumDiff2modes+1):
            index = i - (2*N+1)
            index_for_w = index_2[int(index+1)]
            g0+=2*wavefn_conj[i]*mode_func[index_for_w[1]]*mode_func[index_for_w[0]]*wavefn_save[0]
    return g0

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_1ph(wavefn_save, wavefn_conj):
    g1 = np.zeros(r.size) + 0j
    # Calculate |g,alpha1>
    for i in prange(1, int(N + 1)):
        index = i - 1
        # to |g, alpha1>
        # g1 = g1 + 2 * wavefn_conj[t_index, i] * mode_func[index]**2 * wavefn_save[t_index, i]
        # to |g, beta1>
        for j in prange(1, int(N + 1)):
            index_j = j - 1
            g1 += 2 * wavefn_conj[i] * mode_func[index] * mode_func[index_j] * wavefn_save[j]
    return g1

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_2ph_ii(wavefn_save, wavefn_conj):
    g2 = np.zeros(r.size) + 0j
    g2_ii = np.zeros(r.size) + 0j
    # Calculate |g,alpha2>
    for i in prange(int(N + 1), int(2 * N) + 1):
        index = int(i - (N + 1))
        # to |g, 0>
        g2 += np.sqrt(2) * wavefn_conj[0] * mode_func[index] ** 2 * wavefn_save[i]
        # to |g, alpha2>
        g2_ii += 4 * wavefn_conj[i] * mode_func[index] ** 2 * wavefn_save[i]
        # to |g, alpha1beta1>
        if N != 1:
            for j in prange(2 * int(N) + 1, 2 * int(N) + NumDiff2modes + 1):
                index_j = j - (2 * N + 1)
                index_for_w = index_2[int(index_j + 1)]
                if alpha[index] == alpha[index_for_w[0]]:
                    g2 += 2*np.sqrt(2) * wavefn_conj[j] * mode_func[index]\
                            * mode_func[index_for_w[1]]* wavefn_save[i]
                if alpha[index] == alpha[index_for_w[1]]:
                    g2 += 2*np.sqrt(2) * wavefn_conj[j] * mode_func[index]\
                            * mode_func[index_for_w[0]]* wavefn_save[i]
    return g2+g2_ii

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_2ph_ij(wavefn_save, wavefn_conj):
    g11= np.zeros(r.size) + 0j
    # Calculate |g, alpha1beta1>
    for i in prange(2 * int(N) + 1, 2 * int(N) + NumDiff2modes + 1):
        index = i - (2 * N + 1)
        index_for_w = index_2[int(index + 1)]
        # to |g,0>
        g11 += 2* wavefn_conj[i] * mode_func[index_for_w[0]]\
                     * mode_func[index_for_w[1]]* wavefn_save[0]
        # to |g, alpha2>
        for j in prange(int(N + 1), int(2 * N) + 1):
            index_j = int(j - (N + 1))
            if index_j == alpha[index_for_w[0]]:
                g11 += 2*np.sqrt(2) * wavefn_conj[j] * mode_func[index_for_w[1]] \
                 * mode_func[index_j] * wavefn_save[i]
            if index_j == alpha[index_for_w[1]]:
                g11 += 2*np.sqrt(2) * wavefn_conj[j] * mode_func[index_for_w[0]] \
                 * mode_func[index_j] * wavefn_save[i]
        # to |g, alpha1beta1>
        #g11 = g11 + 2 * wavefn_conj[t_index, i] * mode_func[index_for_w[0]]**2 * wavefn_save[t_index, i]
        #g11 = g11 + 2 * wavefn_conj[t_index, i] * mode_func[index_for_w[1]]** 2 * wavefn_save[t_index, i]
        # to |g, alpha1 i1>
        for j in prange(2 * int(N) + 1, 2 * int(N) + NumDiff2modes + 1):
            index_j = j - (2 * N + 1)
            index_for_w_j = index_2[int(index_j + 1)]
            if alpha[index_for_w[0]] == alpha[index_for_w_j[0]]:
                g11 += 2*wavefn_conj[j] * mode_func[index_for_w[1]] \
                     * mode_func[index_for_w_j[1]] * wavefn_save[i]
            if alpha[index_for_w[0]] == alpha[index_for_w_j[1]]:
                g11 += 2*wavefn_conj[j] * mode_func[index_for_w[1]] \
                     * mode_func[index_for_w_j[0]] * wavefn_save[i]
            if alpha[index_for_w[1]] == alpha[index_for_w_j[0]]:
                g11 += 2*wavefn_conj[j] * mode_func[index_for_w[0]] \
                     * mode_func[index_for_w_j[1]] * wavefn_save[i]
            if alpha[index_for_w[1]] == alpha[index_for_w_j[1]]:
                g11 += 2*wavefn_conj[j] * mode_func[index_for_w[0]] \
                     * mode_func[index_for_w_j[0]] * wavefn_save[i]
    return g11

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_each_timestep(wavefn_save):
    wavefn_conj = np.conjugate(wavefn_save)
    g0 = calc_ele_int_vacuum(wavefn_save, wavefn_conj)
    g1 = calc_ele_int_1ph(wavefn_save, wavefn_conj)
    g2 = calc_ele_int_2ph_ii(wavefn_save, wavefn_conj)
    if N != 1:
        g11 = calc_ele_int_2ph_ij(wavefn_save, wavefn_conj)
    else:
        g11 = np.zeros(r.size)
    ele_intensity = np.real(g0+g1+g2+g11)
    return ele_intensity

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int(wavefn_save):
    all_ele_int = np.zeros((int(tmax / intens_save_t) + 1, r.size))
    for t in prange(int(tmax / intens_save_t) + 1):
        t_index = int(t * intens_save_t / dt / savestep)
        all_ele_int[t] = calc_ele_int_each_timestep(wavefn_save[t_index])
    return all_ele_int

@jit(nopython=True, fastmath=True, parallel=True)
def calc_ele_int_all(wavefn_save, levels):
    if levels == 2:
        ele_intensity = calc_ele_int(wavefn_save[:, :int(wavefn_save.shape[1] / 2)])
        ele_intensity2 = calc_ele_int(wavefn_save[:, int(wavefn_save.shape[1] / 2):])
        all_ele_int = ele_intensity + ele_intensity2
    elif levels == 3:
        ele_intensity = calc_ele_int(wavefn_save[:, :int(wavefn_save.shape[1] / 3)])
        ele_intensity2 = calc_ele_int(wavefn_save[:, int(wavefn_save.shape[1] / 3): int(2*wavefn_save.shape[1] / 3)])
        ele_intensity3 = calc_ele_int(wavefn_save[:, int(2*wavefn_save.shape[1] / 3):])
        all_ele_int = ele_intensity + ele_intensity2 + ele_intensity3
    return all_ele_int

start_time_calc=time.time()

ele_intensity = calc_ele_int_all(wavefn_save, energy.size)

end_time_calc=time.time()
print('Finished calculating field intensity')
print('Running Wall Time = %10.3f second' % (end_time_calc - start_time_calc))

np.savetxt(calcdir + '/ele_intensity.csv', ele_intensity, delimiter=',')

end_time_all=time.time()
print('Calculation Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))

########################################

@jit(nopython=True, fastmath=True)#, parallel=True)
def calc_pho_num(wavefn_save):
    rho=np.abs(wavefn_save)**2
    photon_number_each=np.zeros((int(N), int(tmax / intens_save_t)))
    index_t = 0
    for t in range(int((tmax) / intens_save_t)):
        t_index = int(t * intens_save_t / dt / savestep)
        for i in range(int(N)):
            photon_number_each[i, index_t] = np.sum(rho[t_index, np.where((index[:,1] == i))[0]])\
                                                    + np.sum(rho[t_index, np.where((index[:,2] == i))[0]])
        index_t = index_t + 1
    return photon_number_each

start_time_pho=time.time()

photon_number_each=calc_pho_num(wavefn_save)
np.savetxt(calcdir + '/photon_number_each.csv', photon_number_each, delimiter=',')

end_time_pho=time.time()
print('Calculation of photon number Finished. Running Wall Time = %10.3f second' % (end_time_pho - start_time_pho))

########################################

# @jit(nopython=True, fastmath=True)#, parallel=True)
# def calc_dipolechange(wavefn_save):
#     dipole = np.zeros(wavefn_save.shape[0])
#     for t in range(dipole.size):
#         # print(t, wavefn_save[t, :int(wavefn_save.shape[1]/2)], 'g')
#         # print(t, wavefn_save[t, int(wavefn_save.shape[1]/2):], 'e')
#         #print(np.conj(wavefn_save[t, :int(wavefn_save.shape[1]/2)])*wavefn_save[t, int(wavefn_save.shape[1]/2):])
#         # dipole[t] = mu[0] * 2 * np.real(np.sum(np.conj(wavefn_save[t, :int(wavefn_save.shape[1]/2)]))
#         #                                      *np.sum(wavefn_save[t, int(wavefn_save.shape[1]/2):]))
#         dipole[t] = mu[0] * 2 * np.real(np.sum(np.conj(wavefn_save[t, :int(wavefn_save.shape[1]/2)])
#                                               *wavefn_save[t, int(wavefn_save.shape[1]/2):]))
#     return dipole

# start_time_dipole=time.time()

# #dipole = calc_dipolechange(wavefn_save)
# #np.savetxt('dipole.csv', dipole, delimiter=',')

# end_time_dipole=time.time()
# print('Calculation of dipole Finished. Running Wall Time = %10.3f second' % (end_time_dipole - start_time_dipole))
