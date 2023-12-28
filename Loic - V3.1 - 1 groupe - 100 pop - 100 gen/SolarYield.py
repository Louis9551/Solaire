import math
import numpy as np
from scipy.special import lambertw
from scipy.optimize import minimize_scalar

def LoadTMY(filename):
    # Defines the number of days per month
    dpm = [0,31,28,31,30,31,30,31,31,30,31,30]
    # opens the data dile in read-only mode and stores every line in a buffer
    RawData = open(filename, 'r').readlines()
    # Counts the number of lines and allocate the different tables
    n_lines = len(RawData)
    dt = 1.  # Time step is one hour in TMY files
    n_step = 24
    n_days = 365   # Typical year has alway 365 days
    date = np.arange(0,n_days)
    time = np.arange(0,n_step,dt)
    j = 0 # index for the time
    k = 0 # index for the day
    n_skip = 17 # number of lines to skip at the beginning of the file
    n_bottom = 13 # number of lines to skip at the end of the file

    (tmp, phi_txt) = RawData[0].split(':')
    (tmp, L_txt) = RawData[1].split(':')
    phi = float(phi_txt)
    L = float(L_txt)
    
    # Declares the matrices to store the loaded data
    T2m = np.zeros((n_days,n_step), dtype=float)  # 2-m air temperature (degree Celsius)
    RH = np.zeros((n_days,n_step), dtype=float)   # relative humidity (%)
    Gh = np.zeros((n_days,n_step), dtype=float)   # Global irradiance on the horizontal plane (W/m2)
    Gn = np.zeros((n_days,n_step), dtype=float)   # Beam/direct irradiance on a plane always normal to sun rays (W/m2)
    Dh = np.zeros((n_days,n_step), dtype=float) # Diffuse irradiance on the horizontal plane (W/m2)
    Ir = np.zeros((n_days,n_step), dtype=float) # Surface infrared (thermal) irradiance on a horizontal plane (W/m2)
    WS10m = np.zeros((n_days,n_step), dtype=float) # 10-m total wind speed (m/s)
    WD10m = np.zeros((n_days,n_step), dtype=float) # 10-m wind direction (0 = N, 90 = E) (degree)
    SP = np.zeros((n_days,n_step), dtype=float) # Surface (air) pressure (Pa)
    
    # Run through all lines to read the weather data
    for i in range(n_lines-n_skip-n_bottom):
            (Buf1, Buf2, Buf3, Buf4, Buf5, Buf6, Buf7, Buf8, Buf9, Buf10) = RawData[i+n_skip].split(',')
            (tmp1, tmp2) = Buf1.split(':')
            yy = int(tmp1[0:4])
            mm = int(tmp1[4:6])
            dd = int(tmp1[6:8])
            h = int(tmp2[0:2])
            m = int(tmp2[2:4])
            t_float = float(h)+float(m)/60.
            j, = np.where(time==t_float)
            k = int(dd)-1+np.sum(dpm[0:int(mm)])
            T2m[k,j] = float(Buf2)
            RH[k,j] = float(Buf3)
            Gh[k,j] = float(Buf4)
            Gn[k,j] = float(Buf5)
            Dh[k,j] = float(Buf6)
            Ir[k,j] = float(Buf7)
            WS10m[k,j] = float(Buf8)
            WD10m[k,j] = float(Buf9)
            SP[k,j] = float(Buf10)
    return date, time, L, phi, T2m, RH, Gh, Gn, Dh, Ir, WS10m, WD10m, SP

# Returns the sun position is terms of height and azimuth for a specific day of the year and vector of time step
# specified in LT (legal time)
def SunPosition(d, UTC, L, phi):
    # Seasonal effect as a function of the day index
    beta = 360. * d / 366.
    ET = 0.00002 + 0.4197 * np.cos(np.radians(beta))  - 7.3509 * np.sin(np.radians(beta)) - 3.2265 * np.cos(2*np.radians(beta)) - 9.3912 * np.sin(2*np.radians(beta)) - 0.0903 * np.cos(3*np.radians(beta)) - 0.3361 * np.sin(3*np.radians(beta))
    delta = 23.45 * np.sin ( np.radians(360./365.*(d-81.)))
    # The mean solar time
    TST = UTC + L/15. - ET/60.
    TST = np.where(TST<24, TST, TST-24)
    omega = (TST - 12.) * 15.
    h_n = np.degrees(np.arcsin(np.cos(np.radians(delta - phi))))
    omega_0 = np.degrees(np.arccos(-np.tan(np.radians(phi))*np.tan(np.radians(delta))))
    UTC_0 = -omega_0/15 + 12 - L/15 + ET/60
    UTC_1 = omega_0/15 + 12 - L/15 + ET/60
    # Calculation of height and azimuth
    h = np.degrees(np.arcsin( np.cos(np.radians(delta)) * np.cos(np.radians(omega[:])) * np.cos(np.radians(phi)) + np.sin(np.radians(delta)) * np.sin(np.radians(phi))))
    cosa = (np.cos(np.radians(delta))*np.cos(np.radians(omega))*np.sin(np.radians(phi))-np.sin(np.radians(delta))*np.cos(np.radians(phi)))/np.cos(np.radians(h))
    a = np.where(cosa>=0.,
                 np.degrees(np.arcsin(np.cos(np.radians(delta))*np.sin(np.radians(omega[:]))/np.cos(np.radians(h[:])))),
                 np.degrees(np.pi - np.arcsin( np.cos(np.radians(delta))*np.sin(np.radians(omega[:]))/np.cos(np.radians(h[:])))))
    a = np.where(a < 180., a, a-360.)
    return (a,h,TST, h_n, UTC_0, UTC_1)

def IrradiationRatio(i, gamma, a, h, albedo, Gh, Dh):
    Rs = np.where(h > 0, np.sin(np.radians(i))*np.cos(np.radians(a-gamma)) / np.tan(np.radians(h))+np.cos(np.radians(i)), 0.)
    Rs = np.where(Rs>=0, Rs, 0.)
    Rd = np.where( Dh != 0., (1.+np.cos(np.radians(i)))/2. + (1.-np.cos(np.radians(i)))/2. * albedo * np.divide(Gh,Dh), 0.)
    theta = np.degrees(np.arccos(np.sin(np.radians(i))*np.cos(np.radians(h[:]))*np.cos(np.radians(a[:]-gamma)) + np.cos(np.radians(i))*np.sin(np.radians(h[:]))))
    return Rs, Rd, theta

def GlassAbsorbtion(S, D, theta, n1=1., n2=1.5, e=4., k=3e-3, tau_t=0.95):
    theta_2 = np.degrees(np.arcsin(np.sin(np.radians(theta))*n1/n2))
    rho = np.where(theta != 0, .5*((np.sin(np.radians(theta_2)-np.radians(theta)))**2/(np.sin(np.radians(theta_2)+np.radians(theta)))**2 +(np.tan(np.radians(theta)-np.radians(theta)))**2/(np.tan(np.radians(theta_2)+np.radians(theta)))**2), 0.)
    tau_r = (1.-rho)/(1.+rho)
    tau_a = np.exp(-k*e/np.cos(np.radians(theta_2)))
    tau = np.where(theta<90, tau_a*tau_r, 0.)
    return S*tau+D*tau_t

def PV_cell(V, Gabs, Tamb, NOCT, Rs , Rsh, Id0, T0, Iphi0, G0, n, Eg):
    q = 1.602e-19
    k = 1.38e-23
    Tcell = 273.15 + Tamb + Gabs*(NOCT - 20.)/800.
    Vt = k*Tcell/q
    Id = Id0*(T0/Tcell)**3 * np.exp(q*Eg/n/k*(1./T0 - 1./Tcell))
    Iphi = Iphi0*Gabs/G0
    temp = lambertw(Id*Rs/(n*Vt*(1.+Rs/Rsh))*np.exp(V/n/Vt*(1.-Rs/(Rs+Rsh))+(Iphi+Id)*Rs/(n*Vt*(1.+Rs/Rsh))))
    I = (Iphi + Id - V/Rsh)/(1.+Rs/Rsh) - n*Vt/Rs*temp.real
    return -I*V

def PV_panel(Gabs, Tamb, Scell=15.6*15.6, NOCT=46, Rs = 1., Rsh = 10000., Id0 = 1e-12, T0 = 25+273.15, Iphi0 = 35e-3, G0=1000., n=1., Eg=1.12, ncs=70, ncp=1):
    m = len(Gabs)
    Pm=np.zeros(m, dtype=float)
    Vm=np.zeros(m, dtype=float)
    for i in range(m):
        res = minimize_scalar(PV_cell, args=(Gabs[i], Tamb[i], NOCT, Rs, Rsh, Id0, T0, Iphi0, G0, n, Eg), method='brent', tol=None)
        Pm[i] = -Scell*PV_cell(res.x, Gabs[i], Tamb[i], NOCT, Rs, Rsh, Id0, T0, Iphi0, G0, n, Eg)*ncs*ncp
        Vm[i] = res.x * ncs
    return Pm, Vm

def PV_array(Pm, Vm, Nps, Npp, Prated, V_max_MPPT=450, V_min_MPPT=150, a0=0.01, a1=0.01, a2=0.03, tau_loss=0.1):
    P_in_dc = np.where((Vm*Nps<V_max_MPPT) & ((Vm*Nps>V_min_MPPT)), Pm*Nps*Npp*1e-3, 0.)
    eta_inv = np.where(P_in_dc > 0, 1. - a0 - a1*np.divide(Prated,P_in_dc) - a2*np.divide(P_in_dc,Prated), 0.)
    P_ac = np.where( P_in_dc*eta_inv < 1.2*Prated, P_in_dc*eta_inv*(1-tau_loss), 1.2*Prated)
    return np.where(P_ac>=0., P_ac, 0.)