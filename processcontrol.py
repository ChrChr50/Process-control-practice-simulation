import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# process model
def mixer(x,t,Tf,out):
    y = x[0] # output concentration
    q = 1e-5 # flow rate, m^3/s
    V = 1e-4 # volume, m^3
    P = 20 # power, W
    tau = V/q # residence time, s
    k = 3.3e-9 # intrinsic rate constant
    K = k*P/V # rate constant
    dCadt = q/V*(out-y-K*y*tau)
    return dCadt

# make PID
Tf = 300 # temperature, K
ti = 500 # number of steps in time interval
t = np.linspace(0,ti/10,ti+1) # time, s
dt = t[1]-t[0] # d(time)
e = np.zeros(ti+1) # error
pv = np.zeros(ti+1) # process variable concentration
out = np.zeros(ti+1) # controller output concentration
sp = np.zeros(ti+1) # set point concentration
sp[30:] = 40e3 # set point starting at 30 seconds is 10
y0 = 0.0
der = np.zeros(ti+1) # derivative
integ = np.zeros(ti+1) # integral
P = np.zeros(ti+1) # proportional gain
I = np.zeros(ti+1) # integral gain
D = np.zeros(ti+1) # derivative gain

# PID simple tuning (Dead-time θp --> 0 & τc = 1.0τp)
Kp = 3.0 # proportional gain
Kc = 1.0/Kp # controller gain
tauP = 5.0 # proportional time constant
tauI = tauP # integral time constant
tauD = 0.0 # derivative time constant

# PID tuning (change these values to improve process control)
Kc = Kc*5
tauI = tauI/5
tauD = 1.0

# PID controller
for i in range(0,ti):
    e[i] = sp[i]-pv[i]
    if i >= 1:  # calculate starting on second cycle
        der[i] = (pv[i]-pv[i-1])/dt
        integ[i] = integ[i-1]+e[i]*dt
    P[i] = Kc*e[i]
    I[i] = Kc/tauI*integ[i]
    D[i] = -Kc*tauD*der[i]
    out[i] = out[0]+P[i]+I[i]+D[i]
    y = odeint(mixer,pv[i],[0,dt],args=(Tf,out[i]))
    pv[i+1] = y[-1]
out[ti] = out[ti-1]
integ[ti] = integ[ti-1]
P[ti] = P[ti-1]
I[ti] = I[ti-1]
D[ti] = D[ti-1]

# plot PID response
plt.figure(1,figsize=(15,7))
plt.subplot(2,2,1)
plt.plot(t,sp,'k-',linewidth=2,label='Setpoint (SP)')
plt.plot(t,pv,'r:',linewidth=2,label='Process Variable (PV)')
plt.legend(loc='best')
plt.subplot(2,2,2)
plt.plot(t,P,'g.-',linewidth=2,label=r'Proportional = $K_c \; e(t)$')
plt.plot(t,I,'b-',linewidth=2,label=r'Integral = $\frac{K_c}{\tau_I} \int_{i=0}^{n_t} e(t) \; dt $')
plt.plot(t,D,'r--',linewidth=2,label=r'Derivative = $-K_c \tau_D \frac{d(PV)}{dt}$')    
plt.legend(loc='best')
plt.subplot(2,2,3)
plt.plot(t,e,'m--',linewidth=2,label='Error (e=SP-PV)')
plt.legend(loc='best')
plt.subplot(2,2,4)
plt.plot(t,out,'b--',linewidth=2,label='Controller Output (OP)')
plt.legend(loc='best')
plt.xlabel('time')
plt.show()