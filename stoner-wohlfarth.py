import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



def find_energy_minimum(ku, v, theta, h, ms, initial_guess=3):
    # phi - the angle between magnetisation and external field
    # theta - the angle between external field and easy axis
    # h - the normalized external field given my mu*ms*H/2*Ku
    energy = lambda x : 1/4 - (1/4)*np.cos(2*(x[0]-theta))-h*np.cos(x[0])
    #energy = lambda x : ku*np.sin(x[0])*np.sin(x[0]) - h*ms*np.cos(x[0]-theta)
    res = minimize(energy, initial_guess)

    #print(res)
    
    return res.x

def find_magnetisation(ku, theta, h, ms, initial_guess=3):
    # phi - the angle between magnetisation and external field
    # theta - the angle between external field and easy axis in degrees
    # h - the un-normalized external field given my mu*ms*H/2*Ku
    mu = 1 # the permittivity of air
    h_normal = mu*ms*h/(2*ku)
    #energy = lambda x : 1/4 - (1/4)*np.cos(2*(x[0]-np.deg2rad(theta)))-h*np.cos(x[0])
    energy = lambda x : np.sin(x[0])^2 - mu*h*ms*np.cos(x[0]-theta)
    res = minimize(energy, initial_guess)
    energy_minimum = res.x
    magnetisation = np.cos(energy_minimum)
    #print(res)
    
    return res.x

def simulate_hyst_curve(degrees, initial_guess):
    # degrees - takes the angle between external field and easy axis as degrees
    # initial guess - initial guess for theta in radians
    x = np.linspace(-1,1,100)
    y = list()
    for h in x:
        phi = find_energy_minimum(0.001,1,np.deg2rad(degrees),h,30, initial_guess=initial_guess)
        m = np.cos(phi)
        print(m)
        y.append(m)
        pass
    return x,y

for angle in [90,45,30]:
    
    x,y = simulate_hyst_curve(angle,3.13)
    x1,y1 = simulate_hyst_curve(angle,0.001)
    
    plt.scatter(np.concatenate((x,x1)),np.concatenate((y,y1)), label=f"{angle}")

plt.legend()
plt.show()