import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Process function
def mixer(y, t, Tf, out): 
    q = 1e-5 # flow rate, m^3/s
    V = 1e-4 # volume, m^3
    P = 20 # power, W
    tau = V / q # residence time, s
    k = 3.3e-9 # intrinsic rate constant
    K = k * P / V # rate constant
    dCadt = q / V * (out - y - K * y * tau) # concentration balance for ultrasonic reactor
    
    return dCadt

# Initialize PID arrays
Tf = 300 # temperature, K

ti = 500
t = np.linspace(0, ti / 10, ti + 1) # number of time steps
dt = t[1] - t[0] # time difference

err = np.zeros(ti + 1) # error
pv = np.zeros(ti + 1) # process variable concentration
out = np.zeros(ti + 1) # controller output concentration
sp = np.zeros(ti + 1) # set point concentration
sp[30:] = 40e3 # new set point starting at 30 seconds

der = np.zeros(ti + 1) # derivative
integ = np.zeros(ti + 1) # integral
P = np.zeros(ti + 1) # proportional gain
I = np.zeros(ti + 1) # integral gain
D = np.zeros(ti + 1) # derivative gain

# PID simple tuning (Dead-time θp --> 0 & τc = 1.0τp)
Kp = 2.0 # proportional gain
Kc = 1.0/Kp * 5 # controller gain
tauP = 5.0 # proportional time constant
tauI = tauP / 5 # integral time constant
tauD = 1.0 # derivative time constant

# PID controller
for i in range(0, ti): # iterate through time steps to emulate feedback loop
    err[i] = sp[i] - pv[i]
    if i >= 1:
        der[i] = (pv[i] - pv[i - 1]) / dt # slope
        integ[i] = integ[i - 1] + e[i] * dt # cumulative sum of area under the curve (Riemann sum)
    elif i == 0:
        der[i] = 0
        integ[i] = 0
    P[i] = Kc * e[i] # proportional adjustment for error at time step i
    I[i] = (Kc / tauI) * integ[i] # integral adjustment for error at time step i
    D[i] = -(Kc * tauD) * der[i] # derivative adjustment for error at time step i
    out[i] = out[0] + P[i] + I[i] + D[i] # PID controller
    y = odeint(mixer, pv[i], [0, dt], args = (Tf, out[i])) # solve differential equation for mixer
    pv[i + 1] = y[-1] # make recored process variable for the next time step equal to the most recent diffeq solution
out[ti] = out[ti - 1]
integ[ti] = integ[ti - 1]
P[ti] = P[ti - 1]
I[ti] = I[ti - 1]
D[ti] = D[ti - 1] # save final adjustments

# Plot PID graphs
plt.figure(1, figsize = (15, 7))
plt.subplot(2, 2, 1)
plt.plot(t, sp, 'k-', linewidth = 2, label = 'Setpoint')
plt.plot(t, pv, 'r:', linewidth = 2, label = 'Process Variable')
plt.legend(loc = 'best')
plt.subplot(2, 2, 2)
plt.plot(t, P, 'g.-', linewidth = 2, label = r'Proportional = $K_c \; e(t)$')
plt.plot(t, I, 'b-', linewidth = 2, label = r'Integral = $\frac{K_c}{\tau_I} \int_{i=0}^{n_t} e(t) \; dt $')
plt.plot(t, D, 'r--', linewidth = 2, label = r'Derivative = $-K_c \tau_D \frac{d(PV)}{dt}$')    
plt.legend(loc = 'best')
plt.subplot(2, 2, 3)
plt.plot(t, e, 'm--', linewidth = 2, label = 'Error')
plt.legend(loc = 'best')
plt.subplot(2, 2, 4)
plt.plot(t, out, 'b--', linewidth = 2, label = 'Controller Output')
plt.legend(loc = 'best')
plt.xlabel('time')
plt.show()
