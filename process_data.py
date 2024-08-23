import os 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

dir_path = os.path.dirname(os.path.realpath(__file__))
plt.rcParams['text.usetex'] = True

data_dir_contents = os.listdir(f"{dir_path}\\data\\")

field_error = 0.2 # mT

def load_data(file_path): # Loads the file and returns the data as an array
    res = list()
    with open(file_path, 'r') as fp:
        contents = fp.readlines()
    
    for line in contents[1:]: # Loop through file excluding the first line
        field, grey_level = map(lambda l : l.strip().replace(",","."), line.split("\t")[0:2]) # Parse the data as strings, strip away any whitespace, replace comma with dot
        field, grey_level = (float(field), float(grey_level)) # Convert string data into floating point numbers
        res.append((field, grey_level))
        pass
    return res

def normalize_data(loaded_data): # Normalizes the kerr signal
    x,y = zip(*loaded_data)
    maximum = max(y)
    minimum = min(y)

    y = map(lambda item : 2*(item - minimum)/(maximum - minimum) - 1, y)
    return zip(x,y)

def extract_remnant_magnetization(loop_data, loop_degree, large=False): # Returns the remnant magnetisation of the material
    magnetizations = list() # Append the remnant magentisations to a list as hysterisis graphs are asymmetric
    errors = list()
    eps = 0.1
    ext_field, all_m = zip(*loop_data)

    plt.plot(ext_field,all_m)
    plt.xlabel("External field $H$")
    plt.ylabel("Normalized Kerr signal $\\frac{M}{M_s}$")
    plt.grid(True)
    plt.axvline(x=0, linestyle='dashed', label="x=0")

    closeup_limits = []

    #h,m = zip(*loop_data)
    for count, (ef, m) in enumerate(zip(ext_field, all_m)):
        if abs(ef - 0) < eps:
            min_index = count-1
            max_index = count+1
            x = ext_field[min_index:max_index] # Select a few points from data for regression
            y = all_m[min_index:max_index]

            res = stats.linregress(x,y) # Fit a line
            remanence = res.intercept
            
            hnew = np.linspace(ext_field[min_index], ext_field[max_index], num=10) # New external field H
            mnew = res.slope*hnew + res.intercept # New magnetisation M
            plt.plot(hnew, mnew, label=f"Drawn line y={res.slope:.2f}*x + {res.intercept:.2f}")
            plt.axhline(y=remanence, linestyle='dashed', label="$M_r="+f"{remanence:.2f}$")
            plt.plot([0],[remanence], '.')

            print(f"M_r intercept: {remanence}, epsilon {m}")

            magnetizations.append(m)
            errors.append(res.intercept_stderr)

            # Choose closeup limits
            closeup_x = (ext_field[min_index]*1.1, ext_field[max_index]*1.1)
            y_lim_min = min([all_m[min_index],all_m[max_index]])
            y_lim_max = max([all_m[min_index],all_m[max_index]])
            closeup_y = (y_lim_min-0.1, y_lim_max+0.1)
            closeupxy = (closeup_x, closeup_y)
            closeup_limits.append(closeupxy)
        pass

    save_path = f"{dir_path}\\figures\\remanencences\\remanence_{loop_degree}.png"
    plt.legend()
    plt.savefig(save_path)

    for count, (xlims, ylims) in enumerate(closeup_limits):
        plt.xlim(xlims)
        plt.ylim(ylims)
        save_path1 = f"{dir_path}\\figures\\remanencences_closer\\remanence_{loop_degree}_{count}.png"
        plt.savefig(save_path1)

    plt.clf()

    return magnetizations

def extract_field_coercivity(loop_data, loop_degree, large=False): 
    h,m = zip(*loop_data)
    plt.plot(h,m)
    plt.xlabel("External field $H$")
    plt.ylabel("Normalized Kerr signal $\\frac{M}{M_s}$")
    plt.grid(True)

    coercivities = list() # Append the field coercivities to a list as hysterisis graphs are asymmetric
    all_regression_x, all_regression_y = [], []

    for i in range(len(h)-1):
        if abs(m[i+1] - m[i]) >= 0.5: # If the difference between two magnetisation datapoints are large enough, there has been a sharp turn
            # Now fit a curve between these two points to extrapolate
            x = [h[i], h[i+1]]
            y = [m[i], m[i+1]]

            res = stats.linregress(x,y)

            hnew = np.linspace(h[i], h[i+1], num=100) # New external field H
            mnew = res.slope*hnew + res.intercept # New magnetisation M

            plt.plot(hnew, mnew, label=f"Drawn line y={res.slope:.2f}*x + {res.intercept:.2f}")

            # Using this curve, extract the field coercivity, that is, the external field for which magnetisation is zero
            # Solve the equation 0=k*x + b => x = -b/k

            coercivity = -res.intercept/res.slope # TODO: make nice plots of these fitted curves and coercivities
            plt.axvline(x=coercivity, linestyle='dashed', label="$H_c="+f"{coercivity:.2f}$")

            # TODO: make an error estimate, tansferring the x-erro to y axis
            
            coercivities.append(coercivity) # TODO: estimate some error for this method
            all_regression_x += x
            all_regression_y += y

    plt.axhline(y=0, color='grey', linestyle='dashed', label="$y=0$") # Plot the y=0 line on the graph
    #if not large:
    plt.plot(all_regression_x, all_regression_y, '.', label="Points line drawn through")

    save_path = f"{dir_path}\\figures\\coercivities\\coercivity_{loop_degree}.png"

    if large:
        plt.xlim((-1,1))
        plt.ylim((-0.25,0.25))
        save_path = f"{dir_path}\\figures\\coercivities_closer\\coercivity_{loop_degree}.png"

    plt.legend()

    plt.savefig(save_path)
    plt.clf()
    return coercivities

def extract_field_coercivity_easyaxis(loop_data, loop_degree, large=False): # Use a different approach for extracting coercivites near easy axis
    eps = 0.04 # Use a sufficiently low sensitivity
    res = list()
    h,m = zip(*loop_data)
    plt.plot(h,m, label="Hysteresis curve")
    plt.xlabel("External field $H$")
    plt.ylabel("Normalized Kerr signal $\\frac{M}{M_s}$")
    plt.grid(True)

    m_index, = np.where(np.isclose(m, 0, rtol=1, atol=eps)) # Find the magnetisations that are closet to zero
    all_regression_x, all_regression_y = [], []

    for i in m_index:
        # Fit a line through each such point
        x = [h[i], h[i-1]]
        y = [m[i], m[i-1]]

        lin_res = stats.linregress(x,y)

        coercivity = -lin_res.intercept/lin_res.slope
        res.append(coercivity)

        plt.axvline(x=coercivity, linestyle='dashed', label="$H_c="+f"{coercivity:.2f}$")
        plt.plot(x,y, label=f"Drawn line y={lin_res.slope:.2f}*x + {lin_res.intercept:.2f}")

        all_regression_x += x
        all_regression_y += y
    
    #if not large:
    plt.plot(all_regression_x, all_regression_y, '.', label="Points line drawn through")

    plt.axhline(y=0, color="grey", linestyle="dashed", label="$y=0$")
    
    save_path = f"{dir_path}\\figures\\coercivities\\coercivity_{loop_degree}.png"
    # plt.show()
    if large:
        plt.xlim((-1,1))
        plt.ylim((-0.25,0.25))
        if loop_degree == "90":
            plt.xlim((-0.54, -0.6))
            plt.ylim((-0.025,0.025))
        save_path = f"{dir_path}\\figures\\coercivities_closer\\coercivity_{loop_degree}.png"

    plt.legend()

    plt.savefig(save_path)
    plt.clf()

    return res

def save_plot_loop_separate(loop_data, loop_degree, rems, coers): # Plots all the syteresis curves on separate figures
    x,y = zip(*loop_data)
    plt.grid(True)
    plt.title(f"Sample at {loop_degree}" + "$^{\\circ}$ to field")
    plt.xlabel("Field $H$ (mT)", fontsize=15)
    plt.ylabel("Normalized Kerr signal $\\frac{M}{M_s}$", fontsize=15)
    plt.plot(x,y, linewidth=0.9, color="black", label="Hysteresis")

    # Plot remanceses and coercitiveas as points on a graph
    plt.plot(*zip(*rems), 'x', label="$M_r$", color='red')
    plt.plot(*zip(*coers), 'x', label="$H_c$", color='blue')

    plt.legend()

    plt.savefig(f"{dir_path}\\figures\\hyst-loops\\figure_{loop_degree}.png")
    plt.clf()
    pass

def save_plot_loop_onefig(loop_data, loop_degree): # Plots all the hysteresis loops on a single figure
    x,y = zip(*loop_data)
    plt.grid(True)
    plt.xlabel("Field (mT)")
    plt.ylabel("Gray level")
    plt.plot(x,y, label=loop_degree, linewidth=0.5)
    plt.legend()
    pass
#test_load = load_data(dir_path + r"\data\0_degree\loop_data.txt")
#save_plot_loop(test_load, 0)

data_dir_contents.sort(key=(lambda k : int(k.split("_")[0]) )) # Sort the folders in order of increasing degree

def make_separate_plots(rem_vals, coer_vals): # Makes a separate plot from each hysteresis curve
    for folder_name in data_dir_contents:
        degrees = folder_name.split("_")[0]
        loop_path = f"{dir_path}\\data\\{folder_name}\\loop_data.txt"
        data = load_data(loop_path)

        r1l, r2l = rem_vals[int(degrees)]
        c1l, c2l = coer_vals[int(degrees)]

        print(f"Making plot {int(degrees)} r1: {r1l} r2: {r2l} c1: {c1l}  c2: {c2l}")

        rem_points = [(0,r1l), (0,r2l)]
        coer_points = [(c1l,0), (c2l,0)]

        if degrees == "15":
            # Normalize the 15 degree measurement as its the only one not nomalized
            data = normalize_data(data)

        save_plot_loop_separate(data, degrees, rem_points, coer_points)
        pass

def make_common_plot(): # Plots all the hysteresis curves in one figure
    for folder_name in data_dir_contents:
        degrees = folder_name.split("_")[0]
        loop_path = f"{dir_path}\\data\\{folder_name}\\loop_data.txt"
        data = load_data(loop_path)
        if degrees == "15":
            # Normalize the 15 degree measurement as its the only one not nomalized
            data = normalize_data(data)

        save_plot_loop_onefig(data, degrees)
        if degrees == "90":
            break

        pass
    plt.savefig(f"{dir_path}\\figures\\figure_all_loops.png")

def extract_remnant_magnetisations(): # Returns the remnant magnetisations as function of degree
    res = list()
    for folder_name in data_dir_contents:
        degrees = folder_name.split("_")[0]
        print(f"Processing data for {degrees}")
        loop_path = f"{dir_path}\\data\\{folder_name}\\loop_data.txt"
        data = load_data(loop_path)
        if degrees == "15":
            # Normalize the 15 degree measurement as its the only one not nomalized
            data = normalize_data(data)
        mr = extract_remnant_magnetization(data, degrees) # The remnant magentisation
        # print(mr)
        res.append((int(degrees), mr[0], mr[1]))
    return res

def coervities_degrees(large=False): # Returns the field coercivities as function of degree
    res = list()
    for folder_name in data_dir_contents:
        degrees = folder_name.split("_")[0]
        print(f"Processing data for {degrees}")
        loop_path = f"{dir_path}\\data\\{folder_name}\\loop_data.txt"
        data = load_data(loop_path)
        if degrees == "15":
            # Normalize the 15 degree measurement as its the only one not nomalized
            data = normalize_data(data)
        coer = extract_field_coercivity(data, degrees, large=large)
        print(coer)
        # print(mr)
        try:
            res.append((int(degrees), coer[0], coer[1]))
        except IndexError:
            # If above fails we are near easy axis, different approach must be used
            coer = extract_field_coercivity_easyaxis(data, degrees, large=large)
            print(f"Easy axis {coer}")
            res.append((int(degrees), coer[0], coer[1]))

    return res

def find_magnetisation(ku, theta, h, ms, initial_guess=3):
    # phi - the angle between magnetisation and external field
    # theta - the angle between external field and easy axis in degrees
    # h - the normalized external field given my mu*ms*H/2*Ku
    energy = lambda x : 1/4 - (1/4)*np.cos(2*(x[0]-np.deg2rad(theta)))-h*np.cos(x[0])
    res = minimize(energy, initial_guess)
    energy_minimum = res.x
    magnetisation = np.cos(energy_minimum)
    #print(res)
    
    return magnetisation

def make_remnant_mag_plot():
    deg, mr1, mr2 = zip(*extract_remnant_magnetisations())
    plt.clf()

    plt.plot(deg, mr1, color="blue",marker="x",label="Higher curve $M_r/M_s$")
    plt.plot(deg, mr2, color="red",marker="x", label="Lower curve $Mr/M_s$")

    #x = np.linspace(0,180,10)
    #y = np.cos(np.deg2rad(x))
    #plt.plot(x,y, label="$y=\\cos(\\theta)$")
    #plt.plot(x,-y, label="$y=\\-cos(\\theta)$")

    plt.xlabel("Degrees in Kerr microscope", fontsize=15)
    plt.ylabel("Normalized Kerr signal $\\frac{M_{r}}{M_s}$", fontsize=15)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.grid(True)
    plt.xticks([0,15,30,45,60,75,90,105,120,135,150,165,180])
    plt.savefig(f"{dir_path}\\figures\degrees_remanence.png")
    # plt.show()
    plt.clf()
    return zip(deg, mr1, mr2)

def make_coercivity_plots(large=True):
    # Makes all plots related to coercivity
    coervities_degrees(large=large) # Make closup plots of coercivity extraction
    # Makes and 
    deg, c1, c2 = zip(*coervities_degrees()) # Make regular plots of coercivity extraction
    
    # Make a plot of coercivity as function of degrees on kerr microscope
    plt.errorbar(deg, c1, yerr=0.2, color="blue",marker="x", label="Upper curve", capsize=2)
    #plt.errorbar(deg,c1,yerr=0.2,color="black", linewidth=0.9)

    plt.errorbar(deg, c2, yerr=0.2, color="red",marker="x", label="Lower curve",capsize=2)
    #plt.errorbar(deg,c2,yerr=0.2,color="black", linewidth=0.9)
    
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Degrees in Kerr microscope", fontsize=15)
    plt.ylabel("$H_c$ (mT)", fontsize=15)
    plt.legend()
    plt.xticks([0,15,30,45,60,75,90,105,120,135,150,165,180])
    plt.grid(True)
    plt.xlim(-1,181)
    #plt.show()
    plt.savefig(f"{dir_path}\\figures\degrees_coercivities.png")
    plt.clf()
    return zip(deg, c1, c2)

def make_hard_axis_regression():
    # Makes a linear regression on hard axis hystersis curve for anisotropy
    # constant determination
    hyst = load_data(dir_path + "\\data\\90_degree\\loop_data.txt") # Laod and normalize measurement data along hard axis
    hyst = normalize_data(hyst)

    h,m = zip(*hyst)

    fig, ax = plt.subplots()

    ax.plot(h,m, picker=5, label="Hysteresis curve")
    #ax.title("Hysteresis curve along hard axix (90 degrees in microscope)")

    # def on_pick(event):
    #     line = event.artist
    #     xdata = line.get_xdata()
    #     ydata = line.get_ydata()
    #     ind = event.ind
    #     print(f'on pick line: {xdata[ind]}, {ydata[ind]}')

    # cid = fig.canvas.mpl_connect('pick_event', on_pick)

    ## Select a portion of data between these two points and fit a regression line between them.

    lower_endpoint_h, lower_endpoint_m = (3.023922, 0.90281895) # points both on lower curve
    lower_startpoint_h, lower_startpoint_m = (-2.986704, -0.83661692)

    upper_endpoint_h, upper_endpoint_m = (2.830205,0.86544993)
    upper_startpoint_h, upper_startpoint_m = (-2.982516,-0.8085959)

    lower_starpoint, = np.where(np.isclose(h, lower_startpoint_h))
    lower_endpoint, = np.where(np.isclose(h, lower_endpoint_h))

    upper_starpoint, = np.where(np.isclose(h, upper_startpoint_h))
    upper_endpoint, = np.where(np.isclose(h, upper_endpoint_h))

    print(lower_starpoint[0], lower_endpoint[0], upper_starpoint[0], upper_endpoint[0])

    x = h[lower_starpoint[0]:lower_endpoint[0]] + h[upper_endpoint[0]:upper_starpoint[0]]
    y = m[lower_starpoint[0]:lower_endpoint[0]] + m[upper_endpoint[0]:upper_starpoint[0]]

    res = stats.linregress(x,y)

    newx = np.linspace(-4,4,100)
    k = res.slope
    b = res.intercept
    newy = k*newx + b

    n = len(x)
    helperD = n*sum(map(lambda xi : xi**2,x)) - sum(x)**2
    sigma = 0.2
    deltak = np.sqrt(n*sigma**2/helperD)

    ax.plot(newx,newy, label=f"$y={k:.3f}x + {b:.3f}$")
    ax.grid(True)

    plt.title(f"$\\sigma_k = {res.stderr:.3f}$")

    plt.xlabel("External field $H$", fontsize=15)
    plt.ylabel("Normalized Kerr signal $\\frac{M}{M_s}$", fontsize=15)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.savefig(f"{dir_path}\\figures\\anisotropy-regression-wo-points.png")

    ax.plot(x,y,'.', label="Points selected for regression")
    plt.legend()
    plt.savefig(f"{dir_path}\\figures\\anisotropy-regression.png")

    print(f"y = {k}*x + {b}")
    print(f"N: {n} \n D: {helperD} \n delta k {deltak} \n sigma {0.2}")
    pass

## main
if __name__ == "__main__":
    #make_separate_plots() # Make plots of each hysteresis curve
    #make_common_plot() # Makes a plot of all hysteresis curves on a single figure
    deg_r, r1, r2 = zip(*make_remnant_mag_plot()) # Create the plot about remnant magnetisation against degrees
    #make_coercivity_plots(large=True) # Create the plot about coercivities against degrees
    deg_c, c1, c2 = zip(*make_coercivity_plots(large=False)) # Create small coercivity selection plots

    rems = dict( (x, (y,z)) for x,y,z in zip(deg_r,r1,r2))
    coers = dict( (x, (y,z)) for x,y,z in zip(deg_c,c1,c2))

    print(rems, coers)

    make_separate_plots(rems, coers)
    make_hard_axis_regression()
    pass