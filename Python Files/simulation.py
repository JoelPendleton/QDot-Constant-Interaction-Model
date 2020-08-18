import numpy as np
import matplotlib.pyplot as plt
from random import seed # generates seed for random number generator
from random import random  # random generates a random number between 0 and 1
from random import uniform # generates random float between specified range
from datetime import datetime
import progressbar
import os
import sys


def calculate_current(V_SD, V_G, mu_N):
    e = 1.6E-19
    h = 4.1357E-15
    k_B = 8.6173E-5
    T = 0.01
    R_K = h / (e **2)
    R_T =  1/(  V_G ** 2)


    gammaPlus = np.ones(V_SD.shape)
    gammaNegative = np.ones(V_SD.shape)

    deltaE = mu_N/e -  V_SD
    exponent = deltaE / (k_B * T)
    mask = exponent > 100


    gammaPlus[mask] = -0.01
    mask = exponent < -100
    gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask])

    '''
    condition1 = exponent > -100
    condition2 = exponent < 100
    mask = (condition1 & condition2).astype(int)
    print(mask.shape)

    gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp([mask]))'''

    '''
    if exponent > 100:
        gammaPlus = -0.01
    elif exponent < -100:
        gammaPlus = - (deltaE / h) * (R_K / R_T)
    else:
        gammaPlus = - (deltaE / h) * (R_K / R_T) / (1 - np.exp(exponent))
    '''
    deltaE = mu_N / e
    exponent = -deltaE / (k_B * T)

    mask = exponent > 100
    gammaNegative[mask] = -0.01
    mask = exponent < -100
    gammaNegative[mask] = (deltaE[mask] / h) * (R_K / R_T[mask])


    '''
    condition1 = exponent > -100
    condition2 = exponent < 100
    mask = (condition1 & condition2).astype(int)
    gammaNegative[mask] = (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))'''

    '''   
    if exponent > 100:
        gammaNegative = -0.01
    elif exponent < -100:
        gammaNegative = (deltaE / h) * (R_K / R_T)
    else:
        gammaNegative =  (deltaE / h) * (R_K / R_T) / (1 - np.exp(exponent))
    '''

    current = e * gammaPlus * gammaNegative / (gammaPlus + gammaNegative)
    return current

def electricPotential(n, V_SD_grid, V_G_grid, E_C, N_0, e, C_S, C_G):

    """
    Function to compute the electric potential of the QDot.
    :param n: the number of electrons in the dot
    :param V_SD_grid: the 2d array of source-drain voltage values
    :param V_G_grid: the 2d array of gate voltage values
    :return: The Electric Potential for adding the nth electron to the dot
    """

    E_N = E_C*(((n)**2-(n-1)**2)/n*5+random()/9*n)/(10*n)  # arbitrary random formula used to increase diamond width as more electrons are added

    return E_N, (n - N_0 - 1/2) * E_C - (E_C / e) * (C_S * V_SD_grid + C_G * V_G_grid) + E_N


def currentChecker(mu_N, mu_S, V_SD):
    """
    Function to determne region where current is allowed to flow and where there is a blockade.
    Finds indexes corresponding to values of V_SD and V_G for which current can flow from source-drain or drain-source
    :param mu_N: The electric potential to add the Nth electron to the system.
    :return: The Total allowed current across the grid of voltages. It is either 0, 1, or 2 (units and additive effects of different levels not considered)
    """
    # the algorithm below looks contrived but it removes the need for for loops increasing runtime
    # it checks whether the potential energy of the electron state is between the source and drain
    condition1 = mu_N > 0
    condition2 = mu_N < mu_S
    condition3 = V_SD < 0
    condition4 = mu_N < 0
    condition5 = mu_N > mu_S
    condition6 = V_SD > 0

    # Consider both scenarios where mu_D < mu_N < mu_S and mu_S < mu_N < mu_D
    I_1 = (condition1 & condition2 & condition3).astype(int)
    I_2 = (condition4 & condition5 & condition6).astype(int)
    return I_1 + I_2  # combine the result of these possibilities.



def generate(number_of_examples):
    """
    Function to generate training examples from simulator, calls upon above functions.
    :param number_of_examples: number of simulated examples you wish to generate
    :return: returns True when finished
    """
    # Define Constants
    N = range(1, 11)
    N_0 = 0
    e = 1.6E-19

    # Define a 1D array for the values for the voltages
    # Define a 1D array for the values for the voltages
    V_SD_max = 0.1
    V_G_min = 0.3
    V_G_max = 1
    V_SD = np.linspace(- V_SD_max, V_SD_max, 1000)
    V_G = np.linspace(V_G_min, V_G_max, 1000)

    # Generate 2D array to represent possible voltage combinations

    V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)

    # Define the potential energies of the source and drain

    mu_S = - e * V_SD  # source potential energy


    if not os.path.exists("../Training Data/Training_Input"):
        os.makedirs("../Training Data/Training_Input")
    if not os.path.exists("../Training Data/Training_Output"):
        os.makedirs("../Training Data/Training_Output")



    with progressbar.ProgressBar(max_value=number_of_examples) as bar: # initialise progress bar
        for k in range(1,number_of_examples+1):

            I_tot = np.zeros(V_SD_grid.shape)  # Define the total current
            I_ground = np.zeros(V_SD_grid.shape)  # Define the ground transition current
            E_N_previous = 0  # stores previous E_N value
            V_G_start = 0  # start of current diamond
            diamond_starts = np.zeros(
                (1, len(N)))  # numpy array to store the store positions of each diamond along x-axis

            seed(datetime.now())  # use current time as random number seed

            C_S = 10E-19 * uniform(0.1, 1)  # Uniform used for some random variation
            C_D = 10E-19 * uniform(0.2, 1)
            C_G = 1E-18 * uniform(1, 7)
            C = C_S + C_D + C_G
            E_C = (e ** 2) / C

            plt.figure(figsize=(10,10), dpi = 150)


            for n in N:

                # potential energy of ground to ground transition GS(N-1) -> GS(N)
                E_N, mu_N = electricPotential(n, V_SD_grid, V_G_grid, E_C, N_0, e, C_S, C_G)
                # Indices where current can flow for  GS(N-1) -> GS(N) transitions
                current_ground = calculate_current(V_SD_grid, V_G_grid, mu_N)
                #print(np.max(current_ground))
                #print(np.min(current_ground))

                delta_E_N = E_N - E_N_previous  # Not sure on exact definition yet
                delta_V_G = e/C_G + delta_E_N * C /(e *C_G) # Width of current diamond

                if n ==1:
                    V_G_start = (e/C_G) * (E_N / E_C + 1/2)  # start of first diamond / start of current diamond

                elif n != 1:
                    V_G_start += delta_V_G  # update so start of current diamond
                diamond_starts[0, n-1] = V_G_start

                I_tot += current_ground
                I_ground += current_ground







            I_max = np.max(I_tot)
            I_min = np.min(I_tot)

            # Plot diamonds
            contour = plt.contourf(V_G_grid,V_SD_grid, I_tot, cmap="seismic", levels = np.linspace(I_min,I_max,500)) # draw contours of diamonds
            '''The extra diamonds arose out of the fact that there was a small number of contour levels added in 
            levels attribute to fix this so 0 current was grouped with the small current values '''
            plt.ylim([-V_SD_max, V_SD_max])
            plt.xlim([V_G_min, V_G_max])
            plt.xlabel("$V_{G}$ / V")
            plt.ylabel("$V_{SD}$ / V")

            #plt.title("Single Quantum Dot Coulomb Blockade Simulation")
            cbar = plt.colorbar(contour)
            cbar.set_label("$I$ / A")

            #plt.axis("off")
            #plt.gca().xaxis.set_major_locator(plt.NullLocator()) # trick found on stackex. when trying to get rid of padding
            #plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.savefig("../Training Data/Training_Input/input_{0}.png".format(k))#, bbox_inches='tight', pad_inches=0.0)
            plt.close()





            bar.update(k-1) # update progress bar
    return True






