import numpy as np
import matplotlib.pyplot as plt
import time
from input import *

start_time_all = time.time()
print('Plotting started...')

plot_t=np.arange(0,tmax+dt,dt*savestep)

square_wavefn=np.loadtxt(calcdir + '/rho.csv', delimiter=',')
num_trj=int(square_wavefn[0,0]+square_wavefn[0,1])
pop = square_wavefn/num_trj
Energy=np.loadtxt(calcdir + '/E.csv', delimiter=',')/num_trj
dipole=np.load(calcdir + '/dipole.npz')['dipole']/num_trj
Nph=np.load(calcdir + '/Nph.npz')['Nph']/num_trj
Nphzp=np.load(calcdir + '/Nphzp.npz')['Nphzp']/num_trj
E2=np.load(calcdir + '/E2.npz')['E2']/num_trj
E2zp=np.load(calcdir + '/E2zp.npz')['E2zp']/num_trj
# E0Etall=np.load(calcdir + '/E0Etall.npz')['E0Etall']/num_trj
# Q0Qtall=np.load(calcdir + '/Q0Qtall.npz')['Q0Qtall']/num_trj

#plot
start_time_plot=time.time()

plt.figure()
for i in np.arange(num_atom):
    plt.plot(plot_t,pop[:,2*i], label=r"$ \rho_{\mathrm{g}}$" + "atom " + str(i))
    plt.plot(plot_t,pop[:,2*i+1], '-', label=r"$ \rho_{\mathrm{e}}$" + "atom " + str(i))
    plt.plot(plot_t,pop[:,2*i]+pop[:,2*i+1], label=r"$ \rho_{\mathrm{g}} + \rho_{\mathrm{e}}$"+ "atom" +str(i))
#plt.ylim([0, 1.05])
plt.xlabel(r"$\it t$ (a.u.)")
plt.ylabel(r"$ \rho$")
plt.legend()
plt.tight_layout()
plt.savefig(calcdir + '/Rho.png')
plt.close()

plt.figure()
plt.plot(plot_t, np.sum(Nph, axis=1), label='thermal')
plt.plot(plot_t, np.sum(Nphzp - 0.5, axis=1), label='vaccum')
plt.plot(plot_t, np.sum(Nph, axis=1)+np.sum(Nphzp - 0.5, axis=1), label='all')
plt.xlabel(r"$\it t$ (a.u.)")
plt.ylabel(r"$\langle N_\mathrm{ph} \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig(calcdir + "/Photon_N.png")
plt.close()

plt.figure()
Ec = Energy[:,0] - Energy[0,0]
Eczp = Energy[:,1] - Energy[0,1]
plt.plot(plot_t, Ec, '-.', label='Ec')
plt.plot(plot_t, Eczp, '-.', label='Eczp')
Eq_Eqc_all=np.zeros_like(Ec)
for i in np.arange(num_atom):
    Eq = Energy[:,i+2] - Energy[0,i+2]
    Eqc = Energy[:,i+2+num_atom] - Energy[0,i+2+num_atom]
    plt.plot(plot_t, Eq, label='Eq' +str(i))
    plt.plot(plot_t, Eqc, label='Eqc' +str(i))
    plt.plot(plot_t, Eq + Eqc, '--', label='Eq'+str(i) + ' + Eqc' +str(i))
    Eq_Eqc_all += Eq + Eqc
plt.plot(plot_t, Ec + Eczp + Eq_Eqc_all, label='System E')
plt.xlabel(r"$\it t$ (a.u.)")
plt.ylabel(r"$\langle E \rangle$")
plt.legend()
plt.tight_layout()
#plt.ylim([-0.025,0.025])
plt.savefig(calcdir + "/E.png")
plt.close()


au_to_micron=5.2917724900001E-5
r = np.linspace(0,l,r_resolution)
# zeta = np.sqrt(hbar*w[:,np.newaxis]/(epsilon*l)) * np.sin(alpha[:,np.newaxis]*np.pi/l * r)
# sum_zeta2 = np.sum(zeta**2, axis=0)
for t in range(0, E2.shape[0], int(intens_save_t/(dt*savestep))):
    plot = plt.figure(100+t, figsize=(10,3), dpi=100)
    plt.plot(r * au_to_micron, E2[t], label='thermal, t=' + str(int(t*dt*savestep)))
    plt.plot(r * au_to_micron, E2zp[t], label='vacuum')
    plt.plot(r * au_to_micron, E2[t]+E2zp[t], label='thermal+vacuum')
    for j in np.arange(num_atom):
       plt.vlines(r_atom[j]* au_to_micron, -0.1, 0.1, colors='red', lw=0.5)
    plt.ylim([-0.0001,0.0008])
    plt.xlabel(r"$\it r$ (a.u.)")
    plt.ylabel(r"$\langle I \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(calcdir + "/t"+str(int(t*dt*savestep))+"compare.png")
    plt.close()

for t in range(0, Nph.shape[0], int(intens_save_t/(dt*savestep))):
    plot = plt.figure()
    plt.plot(alpha, Nph[t,:], label='thermal, t=' + str(int(t*dt*savestep)))
    plt.plot(alpha, Nphzp[t,:] - 0.5, label='vacuum')
    plt.plot(alpha, Nph[t,:]+Nphzp[t,:] - 0.5, label='all')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle N_{\mathrm{ph},\alpha} \rangle$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(calcdir + "/Photon_t"+str(int(t*dt*savestep))+".png")
    plt.close()

freq=np.fft.fftfreq(plot_t.size, d=savestep*dt)
dipole_fft=np.fft.fft(dipole[:,0])
plt.figure()
plt.plot(freq * 2*np.pi, np.real(dipole_fft), label='Re')
plt.plot(freq * 2*np.pi, np.imag(dipole_fft), label='Im')
plt.plot(freq * 2*np.pi, np.abs(dipole_fft), label='Abs')
plt.xlabel(r"$\omega$")
plt.ylabel(r"$F[\langle \mu (t) \rangle]$")
plt.legend()
plt.tight_layout()
plt.savefig(calcdir + "/dipole_spectrum.png")
plt.close()

plt.figure()
rfreq=np.fft.rfftfreq(plot_t.size, d=savestep*dt)
E0Et_fft=np.fft.rfft(E0Etall, axis=0)
plt.pcolormesh(r * au_to_micron, rfreq * 2*np.pi, np.abs(E0Et_fft), cmap='hot')
#plt.ylim([0.0, 0.8])
plt.ylabel(r"$F[\langle E(r,0)E(r,t) \rangle]$")
plt.xlabel(r"$\it r$ (a.u.)")
plt.tight_layout()
plt.savefig(calcdir + "/E0Et_spectrum.png")
plt.close()

plt.figure()
rfreq=np.fft.rfftfreq(plot_t.size, d=savestep*dt)
wavefn_fft=np.fft.rfft(pop[:,1])
plt.plot(rfreq * 2*np.pi, np.real(wavefn_fft), label='Re')
plt.plot(rfreq * 2*np.pi, np.imag(wavefn_fft), label='Im')
plt.plot(rfreq * 2*np.pi, np.abs(wavefn_fft), label='Abs')
plt.xlabel(r"$\omega$")
plt.ylabel(r"$F[\langle \rho (t) \rangle]$")
plt.legend()
#plt.xlim([0, 0.8])
plt.tight_layout()
plt.savefig(calcdir + "/wavefn_spectrum.png")
plt.close()

Q0Qt_fft=np.fft.rfft(Q0Qtall, axis=0)
if N==1:
    plt.figure()
    plt.plot(rfreq * 2*np.pi, np.abs(Q0Qt_fft))
    plt.ylabel(r"$F[\langle Q_\alpha (0) Q_\alpha (t) \rangle]$")
    plt.xlabel(r"$\omega$")
    plt.tight_layout()
else:
    plt.figure()
    plt.pcolormesh(alpha, rfreq * 2*np.pi, np.abs(Q0Qt_fft), cmap='hot')
    plt.ylabel(r"$F[\langle Q_\alpha (0) Q_\alpha (t) \rangle]$")
    plt.xlabel(r"$\alpha$")
    plt.tight_layout()
plt.savefig(calcdir + "/Q0Qt_spectrum.png")
plt.close()

end_time_all=time.time()
print('Plotting Finished. Running Wall Time = %10.3f second' % (end_time_all - start_time_all))
