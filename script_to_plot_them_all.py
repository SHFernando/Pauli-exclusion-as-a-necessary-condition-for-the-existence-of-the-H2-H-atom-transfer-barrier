#PLEASE COPY ALL THE .csv AND orb{i}.txt FILES TO THE EXECUTION DIRECTORY

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib import font_manager
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib import colors
from scipy.interpolate import make_smoothing_spline

font_dirs = ["$PATH_TO_FONT_DIR.fonts"]  # The path to the custom font file. COMMENT OUT THESE IF NOT NEEDED
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)#COMMENT OUT THESE IF NOT NEEDED

for font_file in font_files:#COMMENT OUT THESE IF NOT NEEDED
    font_manager.fontManager.addfont(font_file)#COMMENT OUT THESE IF NOT NEEDED

plt.rcParams['font.family'] = 'sans-serif'#COMMENT OUT THESE IF NOT NEEDED
plt.rcParams['font.sans-serif'] = ['Liberation Sans']#COMMENT OUT THESE IF NOT NEEDED


factor=2625.49963947919#Ha to kJ/mol conversion factor derived from NIST data


def stacked_2x1():
    print('Plotting stacked doublet PES and KE surfaces')
    plot_energy = 'total energy'
    plot_energy2 = 'kinetic energy'

    data1 = pd.read_csv('BL_energies_doublet_MEPtrace_cas.csv')
    data2 = pd.read_csv('BL_energies_singlet_MEPtrace_cas.csv')
    data3 = pd.read_csv('BL_energies_triplet_MEPtrace_cas.csv')

    xd = data1['x'].to_numpy()-data1['y'].to_numpy()
    xs = data2['x'].to_numpy()-data2['y'].to_numpy()
    xt = data3['x'].to_numpy()-data3['y'].to_numpy()
    doub_te = data1[plot_energy].to_numpy()
    sing_te = data2[plot_energy].to_numpy()
    trip_te = data3[plot_energy].to_numpy()
    doub_ke = data1[plot_energy2].to_numpy()
    sing_ke = data2[plot_energy2].to_numpy()
    trip_ke = data3[plot_energy2].to_numpy()
    norm_doub2 = 1.1492085+0.49930588401532#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom
    norm_sing2 = 1.1492090#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom
    norm_trip2 = 1.0628984#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom
    norm_doub = -1.1516021992-0.49982118#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom
    norm_sing = -1.1516021843#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom
    norm_trip = -1.0443830553#CAS(2,2) optimized H2 and 100 Angstrom sepated H atom

    ndoub_te = (doub_te-norm_doub)*factor
    nsing_te = (sing_te-norm_sing)*factor
    ntrip_te = (trip_te-norm_trip)*factor
    ndoub_ke = (doub_ke-norm_doub2)*factor
    nsing_ke = (sing_ke-norm_sing2)*factor
    ntrip_ke = (trip_ke-norm_trip2)*factor

    interp_d = CubicSpline(xd,ndoub_te)
    interp_s = CubicSpline(xs,nsing_te)
    interp_t = CubicSpline(xt,ntrip_te)
    interp_dk = CubicSpline(xd,ndoub_ke)
    interp_sk = CubicSpline(xs,nsing_ke)
    interp_tk = CubicSpline(xt,ntrip_ke)

    fig, ((ax1),(ax2)) = plt.subplots(2, 1, sharex='col',sharey='row',figsize=(8,12),dpi=300)
    xd_int = np.arange(min(xd),max(xd),0.00001)
    xs_int = np.arange(min(xs),max(xs),0.00001)
    xt_int = np.arange(min(xt),max(xt),0.00001)
    ax2.set_xlabel(r'$r_1-r_2\,/\,\mathrm{\AA}$', labelpad=12, fontsize=20)
    ax2.set_ylabel(r'${\Delta}E\,/\,$kJ$\,$mol$^{-1}$', labelpad=12, fontsize=20)
    ax1.set_ylabel(r'${\Delta}T\,/\,$kJ$\,$mol$^{-1}$', labelpad=12, fontsize=20)

    ax1.plot(xt_int,interp_tk(xt_int), '--', label='Triplet',color='red', lw=1.2)
    ax1.plot(xd_int,interp_dk(xd_int), '--', label='Doublet',color='blue', lw=1.2)
    ax1.plot(xs_int,interp_sk(xs_int), '--', label='Singlet',color='green', lw=1.2)

    ax2.plot(xd_int,interp_d(xd_int), '-', label='Doublet',color='blue', lw=1.2)
    ax2.plot(xt_int,interp_t(xt_int), '-', label='Triplet',color='red', lw=1.2)
    ax2.plot(xs_int,interp_s(xs_int), '-', label='Singlet',color='green', lw=1.2)

    fig.subplots_adjust(hspace=0.03)
    kwargs = dict(marker=[(-1, 0), (1, 0)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
    ax2.plot(0,1, transform=ax2.transAxes, **kwargs)
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    #ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    ax1.spines.bottom.set_visible(False)
    ax1.spines.bottom.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax1.tick_params(bottom=False)
    ax1.tick_params(which='major', labelsize=18)
    ax2.tick_params(which='major', labelsize=18)
    ax1.legend(fontsize=16)
    ax2.legend(fontsize=16)

    plt.savefig('2x1normalized_stack.png',bbox_inches='tight')


def eda():
    print('Plotting EDA surfaces and DFT KE gradient')
    data1 = pd.read_csv("eda_wb97mv.csv")
    strain = False
    frgE = False
    x1 = data1['x'].to_numpy()
    y1 = data1['y'].to_numpy()

    frz = data1['frozen'].to_numpy()
    pol = data1['pol'].to_numpy()
    ct = data1['ct'].to_numpy()
    tot = data1['total'].to_numpy()
    diff = x1-y1
    cutoff = np.where(diff==0)
    i = int(cutoff[0][0])
    j = None
    cumulative1 = frz + pol
    cumulative2 = cumulative1 + ct
    frag0_eq = -1.16135028#wB97M-V caculated isolated H2 fragment energy
    frag1_eq = -0.49432970#wB97M-V caculated isolated H atom fragment energy
    items_frz_decomp=['pauli','clas_pauli','elec','clas_elec']
    dat_lst = [frz,pol,ct,cumulative1,cumulative2]
    palette = ['red','green','blue', 'brown', 'black']

    if os.path.exists('BL_energies_MEPtrace_wb97mv.csv'):
        strain = True
        data2 = pd.read_csv('BL_energies_MEPtrace_wb97mv.csv')
        x1_adsurf = data2['x'].to_numpy()
        y1_adsurf = data2['y'].to_numpy()
        adjusted = False
        
        if len(x1_adsurf) != len(x1):
            print('arrays are not equal')
            bl_id = 0
            print('removing non-matched entries')
            adjusted = True
            while True:
                nonmatched = False
                x1_adsurf = data2['x'].to_numpy()
                y1_adsurf = data2['y'].to_numpy()
                try:
                    if x1[bl_id] != x1_adsurf[bl_id] and y1[bl_id] != y1_adsurf[bl_id]:
                        nonmatched = True

                except IndexError:
                    break
                
                if nonmatched:
                    data2.drop(bl_id,inplace=True)
                    data2 = data2.reset_index(drop=True)
                    print(f'removing BL_energies.csv row {bl_id} and resetting indices')
                else:
                    bl_id += 1
        if adjusted:
            print('saving adjusted_BL_energies as .csv')
            data2.to_csv('adjusted_BL_energies.csv')
        x1_adsurf = data2['x'].to_numpy()
        y1_adsurf = data2['y'].to_numpy()
        if len(x1_adsurf) != len(x1):
            print('arrays are not equal')
            sys.exit(1)
        if len(x1_adsurf) == len(x1):
            print('arrays are equal')
        total_E_ha = data2['total energy'].to_numpy()
        k_E_ha = data2['kinetic energy'].to_numpy()
        total_E = (total_E_ha-(-0.49432970-1.16135028))*factor
        k_E = (k_E_ha-(1.13683188557732+0.49289557508978))*factor
        if not os.path.exists('H2_relaxed.csv'):
            strain_E = total_E - tot
            strain_E_frm_frag = (data1['frag_0_E']+data1['frag_1_E']-frag0_eq-frag1_eq)*factor
            datframe = pd.DataFrame({'GD': strain_E,'GD_frag': strain_E_frm_frag})
            datframe.to_csv('calc_strainE.csv')
    
    if strain:
        #Calculate kinetic energy gradient with central differences
        KE_grad_SW_id=None
        if os.path.exists('BL_energies_MEPtrace_wb97mv.csv'):
            ke=data2['kinetic energy'].to_numpy()[::-1]
            kex=data2['x'].to_numpy()[::-1]-data2['y'].to_numpy()[::-1]

            delta_KE = ke[2:]-ke[:-2]
            delta_RC = kex[2:]-kex[:-2]
            KE_grad = delta_KE/delta_RC
            itp_grad=make_smoothing_spline(kex[1:-1],KE_grad,lam=1e-2)
            smooth_grad=itp_grad(kex)
            KE_grad_SW_id=np.where(smooth_grad==max(smooth_grad[:int(len(smooth_grad)/2)]))[0][0]


            fig,ax = plt.subplots(figsize=(8,6), dpi=300)
            ax.plot(kex[1:-1],KE_grad, color='k', linestyle='-', lw=1.0,label=r'dT$\,/\,$d(r$_1-$r$_2$)')
            ax.plot(kex,smooth_grad, color='tab:orange', linestyle='-', lw=1.0,label=r'Smooth dT$\,/\,$d(r$_1-$r$_2$)')
            ax.set_xlabel(r'$r_1-r_2\,/\,\mathrm{\AA}$',fontsize=20)
            ax.set_ylabel(r'Gradient / Ha$\,\mathrm{\AA}^{-1}$',fontsize=20,labelpad=10)
            ax.axvline(kex[KE_grad_SW_id],linestyle='--',color='k',lw=0.8)
            print(f'Kinetic energy inflection point calculated at x={kex[KE_grad_SW_id]}')
            yformatter = FuncFormatter(lambda value, _: f'{value:.2f}')
            ax.yaxis.set_major_formatter(yformatter)
            ax.tick_params(axis='both', which='major', labelsize=18)
            formatter = FuncFormatter(lambda value, _: f'{value:.2f}')
            ax.xaxis.set_major_formatter(formatter)
            ax.set_aspect('auto')
            ax.legend(fontsize=16)
            plt.savefig('wb97mv_KE_grad.png',bbox_inches='tight')

        #Plot the overlay figure
        fig, ((ax1),(ax2)) = plt.subplots(2, 1, sharex='col',sharey='row',figsize=(8,12),dpi=300)
        ax1.plot(diff[i:j],frz[i:j], color=palette[0], linestyle='dashdot', lw=1.2,label=r'${\Delta}E_\mathrm{FRZ}$')
        ax1.plot(diff[i:j],cumulative1[i:j], color=palette[3], linestyle='--', lw=1.2,label=r'${\Delta}E_\mathrm{FRZ}+{\Delta}E_\mathrm{POL}$')
        ax1.plot(diff[i:j],cumulative2[i:j], color=palette[4], linestyle=(0,(8,6)), lw=1.2,label=r'${\Delta}E_\mathrm{FRZ}+{\Delta}E_\mathrm{POL}+{\Delta}E_\mathrm{CT}$')
        ax1.plot(diff[i:j],strain_E[i:j], color='navy', linestyle=(0, (4, 6, 8, 3)), lw=1.8,label=r'${\Delta}E_\mathrm{GD}$')
        ax1.plot(diff[i:j],k_E[i:j], color='k', linestyle='-', lw=1.8,label=r'${\Delta}T$')
        print(f'Kinetic energy max point calculated at x={diff[i:j][np.argwhere(k_E[i:j]==max(k_E[i:j]))]}')
        ax1.axvline(diff[i:j][np.argwhere(k_E[i:j]==max(k_E[i:j]))],ymin=0.15,linestyle=(0,(5,2)),color='k',lw=0.8)
        ax1.text(diff[i:j][np.argwhere(k_E[i:j]==max(k_E[i:j]))][0][0]+0.01,150,r'$\frac{dT}{d\mathbf{R}}=0$',fontsize=20)
        if KE_grad_SW_id:
            ax1.axvline(diff[i:j][KE_grad_SW_id],ymin=0.15,linestyle=(0,(5,2)),color='k',lw=0.8)
            ax1.text(diff[i:j][KE_grad_SW_id]-0.4,95,r'$\frac{d^2T}{d\mathbf{R}^2}=0$',fontsize=20)
        ax1.axhline(0,linestyle='-',color='k',lw=0.8)
        ax1.set_ylabel(r'Energy change$\,/\,$kJ$\,$mol$^{-1}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.0f}')
        ax1.yaxis.set_major_formatter(yformatter)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.set_aspect('auto')
        ax1.legend(fontsize=16)
        ax2.plot(diff[i:j],data1['pauli'][i:j], color='k', linestyle='--', lw=1.2,label=r'Pauli')
        ax2.plot(diff[i:j],data1['elec'][i:j], color='k', linestyle='dashdot', lw=1.2,label=r'ELEC')
        ax2.plot(diff[i:j],data1['disp'][i:j], color='k', linestyle=(0,(8,6)), lw=1.2,label=r'DISP')
        ax2.set_xlabel(r'$r_1-r_2\,/\,\mathrm{\AA}$',fontsize=20)
        ax2.set_ylabel(r'Energy change$\,/\,$kJ$\,$mol$^{-1}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.0f}')
        ax2.yaxis.set_major_formatter(yformatter)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        formatter = FuncFormatter(lambda value, _: f'{value:.1f}')
        ax2.xaxis.set_major_formatter(formatter)
        ax2.set_aspect('auto')
        ax2.legend(fontsize=16)
        ax1.spines.bottom.set_visible(False)
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.spines.right.set_visible(False)
        ax1.tick_params(bottom=False)
        fig.subplots_adjust(hspace=0.03)
        kwargs = dict(marker=[(-1, 0), (1, 0)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
        ax2.plot(0,1, transform=ax2.transAxes, **kwargs)
        plt.savefig('2x1EDA_KE_GD_frz_decomp.png',bbox_inches='tight')

def natorb():
    
    if os.path.exists('cas_doublet_NO_KE.csv') and os.path.exists('cas_doublet_NO_occ.csv'):
        print('Plotting NOs and NO KE')
        data1 = pd.read_csv('cas_doublet_NO_KE.csv')
        data2 = pd.read_csv('cas_doublet_NO_occ.csv')
        x = data1['r1-r2']
        orbitals = []
        i = 1
        while True:
            label = f'NO{i}'
            if label in data1.columns:
                orbitals.append(label)
                i += 1 
            else:
                break

        orb_occ = {}
        orb_ke = {}
        orb_ke_1e = {}

        for key in orbitals:
            orb_ke[key] = data1[key]
            orb_occ[key] = data2[key]

        for key in orbitals:
            orb_ke_1e[key] = orb_ke[key]/orb_occ[key]

        #reference Natural orbital one-electron KE values, KE contribution to total KE, and occupation. comma separated
        #list is orbital1, orbital2 and orbital3 respectively
        ref = [(1.1130397/1.97490170),(0.4993059/1.00000000),(0.0361692/0.02509830)]
        ref_OKE = [(1.1130397),(0.4993059),(0.0361692)]
        ref_occ = [(1.97490170),(1.00000000),(0.02509830)]

        linstyles = ['-','--','dashdot']

        #Plot the overlay of normalized 1 electron orbital kinetic energies
        fig, ((ax1),(ax2),(ax3)) = plt.subplots(3, 1, sharex='col',sharey='row',figsize=(8,12),dpi=300,
        gridspec_kw={'height_ratios':[0.5,0.7,1]})
        for i,key in enumerate(orbitals):
            ax2.plot(x,orb_ke_1e[key]*factor-ref[i]*factor, color='k',label=fr'$\phi_{i+1}$', linestyle=linstyles[i], lw=1.2)
        ax2.set_ylabel(r'${\Delta}\langle\phi_i\vert\hat{T}\vert\phi_i\rangle\,/\,$kJ$\,$mol$^{-1}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.0f}')
        ax2.yaxis.set_major_formatter(yformatter)
        ax2.tick_params(labelsize=16)
        ax2.legend(fontsize=16)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(150))
        totalKE = np.zeros_like(orb_ke[next(iter(orb_ke))])
        for i,key in enumerate(orbitals):
            totalKE = totalKE+orb_ke[key]-ref_OKE[i]
            ax3.plot(x,orb_ke[key]*factor-ref_OKE[i]*factor, color='k',label=fr'$T_{i+1}={{\eta}}_{i+1}\langle\phi_{i+1}\vert\hat{{T}}\vert\phi_{i+1}\rangle$', linestyle=linstyles[i], lw=1.2)
        ax3.plot(x,totalKE*factor, color='k',label=r'$T=\sum_{i=1}^{3}T_\mathrm{i}$', linestyle=(0,(7,5,4,8)), lw=1.2)
        ax3.set_xlabel(r'$r_1-r_2\,/\,\mathrm{\AA}$',fontsize=20)
        ax3.set_ylabel(r'${\Delta}T_\mathrm{i}\,/\,$kJ$\,$mol$^{-1}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.0f}')
        ax3.yaxis.set_major_formatter(yformatter)
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(150))
        ax3.tick_params(labelsize=16)
        ax3.legend(fontsize=16)
        for i,key in enumerate(orbitals):
            ax1.plot(x,orb_occ[key], color='k',label=fr'$\eta_{i+1}$', linestyle=linstyles[i], lw=1.2)
        ax1.set_ylabel(r'${\eta_i}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.1f}')
        ax1.yaxis.set_major_formatter(yformatter)
        formatter = FuncFormatter(lambda value, _: f'{value:.1f}')
        ax1.xaxis.set_major_formatter(formatter)
        ax1.tick_params(labelsize=16)
        ax1.legend(fontsize=16,loc='upper right', bbox_to_anchor=(0.98, 0.95))
        ax1.spines.bottom.set_visible(False)
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.spines.right.set_visible(False)
        ax1.tick_params(bottom=False)
        ax2.spines.bottom.set_visible(False)
        ax3.spines.top.set_visible(False)
        ax3.spines.right.set_visible(False)
        ax2.tick_params(bottom=False)
        fig.subplots_adjust(hspace=0.03)
        kwargs = dict(marker=[(-1, 0), (1, 0)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
        ax2.plot(0,1, transform=ax2.transAxes, **kwargs)
        ax2.plot(0,0, transform=ax2.transAxes, **kwargs)
        ax3.plot(0,1, transform=ax3.transAxes, **kwargs)
        plt.savefig('overlay_orb.png',bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(8,6),dpi=300)
        for i,key in enumerate(orbitals):
            ax.plot(x,orb_occ[key]-ref_occ[i], color='k',label=fr'$\eta_{i+1}$', linestyle=linstyles[i], lw=1.2)
        ax.set_xlabel(r'$r_1-r_2\,/\,\mathrm{\AA}$',fontsize=20)
        ax.set_ylabel(r'${\Delta}{\eta_i}$',fontsize=20,labelpad=10)
        yformatter = FuncFormatter(lambda value, _: f'{value:.3f}')
        ax.yaxis.set_major_formatter(yformatter)
        formatter = FuncFormatter(lambda value, _: f'{value:.1f}')
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(labelsize=16)
        plt.legend(fontsize=16)
        plt.savefig('overlay_occ.png',bbox_inches='tight')

def stacked_3x3_amp():
    #Subplots are arrenged top 1-4-7 to bottom 3-6-9
    #Q-Chem .fchk files were processed using multiwavefunction software. 
    #Multiwavefunction probability amplitudes are in the range -3.5 to 3.5 Angstrom along the z-axis of H3
    #three geometries of H3 are represented by columns and orbitals by rows
    #data files 1-3, 4-6 and 7-9 represent the set of three orbital data per-column
    et='total energy'

    dictx = {}
    dicty = {}
    yranges = {}
    i = 1

    while True:
        if os.path.exists(f'orb{i}.txt'):
            with open(f'orb{i}.txt') as f:
                x=np.array([])
                y=np.array([])
                lines = f.readlines()
                key = f'orb{i}'
                for l in lines:
                    x=np.append(x,float(l.split()[3]))
                    y=np.append(y,float(l.split()[4]))

                dictx[key] = x
                dicty[key] = y
                i += 1

        else:
            break
    if i==10:
        print('plotting probability amplitudes')
    else:
        print('expecting 9 orbital data files, skip plotting probability amplitudes')
    i=1

    for key in dicty:
        yranges[f'ax{i}_range'] = max(dicty[key])-min(dicty[key])
        i += 1

    ax1_range = max([yranges['ax1_range'].tolist()]+[yranges['ax4_range'].tolist()]+[yranges['ax7_range'].tolist()])
    ax2_range = max([yranges['ax2_range'].tolist()]+[yranges['ax5_range'].tolist()]+[yranges['ax8_range'].tolist()])
    ax3_range = max([yranges['ax3_range'].tolist()]+[yranges['ax6_range'].tolist()]+[yranges['ax9_range'].tolist()])

    fig, ((ax1, ax4,ax7),(ax2,ax5,ax8),(ax3,ax6,ax9)) = plt.subplots(3, 3, sharex='col',sharey='row',
    figsize=(16,16),dpi=300,
    gridspec_kw={'height_ratios':[ax3_range/ax1_range,ax2_range/ax1_range,ax1_range/ax1_range]})

    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    x_mid_norm3 = np.linspace(-3.5,3.5,len(dictx['orb3']))
    x_mid_norm2 = np.linspace(-3.5,3.5,len(dictx['orb2']))
    x_mid_norm1 = np.linspace(-3.5,3.5,len(dictx['orb1']))
    ax3.plot(x_mid_norm3,dicty['orb1'], '-',color='k')
    ax2.plot(x_mid_norm2,dicty['orb2'],'-',color='k')
    ax1.plot(x_mid_norm1,dicty['orb3'],'-',color='k')

    ax4.plot(x_mid_norm3,dicty['orb6'],'-',color='k')
    ax5.plot(x_mid_norm2,dicty['orb5'],'-',color='k')
    ax6.plot(x_mid_norm1,dicty['orb4'],'-',color='k')

    ax7.plot(x_mid_norm1,dicty['orb9'],'-',color='k')
    ax8.plot(x_mid_norm1,dicty['orb8'],'-',color='k')
    ax9.plot(x_mid_norm1,dicty['orb7'],'-',color='k')

    axes1 = [ax1,ax2,ax4,ax5,ax7,ax8]
    for ax in axes1:
        ax.spines.bottom.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.tick_params(bottom=False)
    ax3.spines.top.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax6.spines.top.set_visible(False)
    ax6.spines.right.set_visible(False)
    ax9.spines.top.set_visible(False)
    ax9.spines.right.set_visible(False)

    fig.subplots_adjust(hspace=0.05,wspace=0.05)
    #ax1.set_ylim(bottom=ax1_min,top=ax1_max , auto=False)
    #ax2.set_ylim(bottom=ax2_min,top=ax2_max , auto=False)
    #ax3.set_ylim(bottom=ax3_min,top=ax3_max , auto=False)

    kwargs = dict(marker=[(-1, 0), (1, 0)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
    ax2.plot([0,0],[0,1], transform=ax2.transAxes, **kwargs)
    ax3.plot(0, 1, transform=ax3.transAxes, **kwargs)
    ax1.set_ylabel(r'${\phi}_3$',fontsize=34,labelpad=18)
    ax2.set_ylabel(r'${\phi}_2$',fontsize=34,labelpad=18)
    ax3.set_ylabel(r'${\phi}_1$',fontsize=34,labelpad=18)
    ax3.set_xlabel(r'z $\,/\,\mathrm{\AA}$',fontsize=30)
    ax6.set_xlabel(r'z $\,/\,\mathrm{\AA}$',fontsize=30)
    ax9.set_xlabel(r'z $\,/\,\mathrm{\AA}$',fontsize=30)
    ax3.tick_params(which='major', labelsize=30)
    ax6.tick_params(which='major', labelsize=30)
    ax9.tick_params(which='major', labelsize=30)
    ax2.tick_params(which='major', labelsize=30)
    ax1.tick_params(which='major', labelsize=30)

    ax4.spines.left.set_visible(False)
    ax5.spines.left.set_visible(False)
    ax6.spines.left.set_visible(False)
    ax7.spines.left.set_visible(False)
    ax8.spines.left.set_visible(False)
    ax9.spines.left.set_visible(False)
    ax6.tick_params(left=False)
    ax5.tick_params(left=False)
    ax4.tick_params(left=False)
    ax9.tick_params(left=False)
    ax8.tick_params(left=False)
    ax7.tick_params(left=False)

    atom_pos = [-0.7552087201092559,0.0,2.4959842047537606]
    atom_pos_1 = [-0.7676677342171011,0.0,1.4811879649283832]
    atom_pos_2 = [-0.956683,0.0,0.956718]

    for plt_no,ax in enumerate(axes):
        if plt_no > 5:
            for pos in atom_pos_2:
                ax.axvline(x=pos,linestyle='--',linewidth=2.0,color=(0,0,0),alpha=0.5)
            ax.axhline(y=0.0,linestyle='--',linewidth=1.4,color='k')
            ax.plot(atom_pos_2[0],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos_2[1],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos_2[2],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        elif plt_no > 2 and plt_no < 6:
            for pos in atom_pos_1:
                ax.axvline(x=pos,linestyle='--',linewidth=2.0,color=(0,0,0),alpha=0.5)
            ax.axhline(y=0.0,linestyle='--',linewidth=1.4,color='k')
            ax.plot(atom_pos_1[0],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos_1[1],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos_1[2],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        else:
            for pos in atom_pos:
                ax.axvline(x=pos,linestyle='--',linewidth=2.0,color=(0,0,0),alpha=0.5)
            ax.axhline(y=0.0,linestyle='--',linewidth=1.4,color='k')
            ax.plot(atom_pos[0],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos[1],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.plot(atom_pos[2],0,linestyle='',color='tab:blue',marker='.',markersize=24)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    #ax1.text(-3.5,0.3,r'$T=1.448\,$Ha',fontsize=15)
    ax2.text(-3.5,0.3,r'$T=0.563\,$',fontsize=20)
    ax3.text(-3.6,0.3,r'$T=0.505\,$',fontsize=20)
    #ax4.text(-3.5,0.3,r'$T=1.517\,$Ha',fontsize=15)
    ax5.text(-3.5,0.3,r'$T=0.556\,$',fontsize=20)
    ax6.text(-3.6,0.3,r'$T=0.549\,$',fontsize=20)
    #ax7.text(-3.5,0.3,r'$T=1.419\,$Ha',fontsize=15)
    ax8.text(-3.5,0.3,r'$T=0.627\,$',fontsize=20)
    ax9.text(-3.6,0.3,r'$T=0.479\,$',fontsize=20)
    ax1.set_title(r'$r_1-r_2=−1.74\,\mathrm{\AA}$',pad=14,fontsize=24)
    ax4.set_title(r'$r_1-r_2=−0.71\,\mathrm{\AA}$',pad=14,fontsize=24)
    ax7.set_title(r'$r_1-r_2=0.0\,\mathrm{\AA}$',pad=14,fontsize=24)

    plt.savefig('stacked.png',bbox_inches='tight',pad_inches=0.5)

if __name__=='__main__':
    stacked_2x1()
    eda()
    natorb()
    stacked_3x3_amp()