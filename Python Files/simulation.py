import numpy as np
import matplotlib.pyplot as plt
from random import seed # generates seed for random number generator
from random import random  # random generates a random number between 0 and 1
from random import uniform # generates random float between specified range
from random import randint
from datetime import datetime
import progressbar
import os


def calculate_current(V_SD, V_G, mu_N, positive):

    e = 1.6E-19
    h = 4.1357E-15
    k_B = 8.6173E-5
    T = 0.01
    R_K = h / (e **2)
    R_T =  1/( 1E-26 * V_G ** 2)

    gammaPlus = np.ones(V_SD.shape)
    gammaNegative = np.ones(V_SD.shape)

    if positive: # If current is positive: flows from source -> drain

        'compute gammaPlus'
        deltaE = mu_N/e +  V_SD
        exponent = deltaE / (k_B * T)

        # if exponent is very positive
        mask = exponent > 100
        gammaPlus[mask] = -0.01 # make gammaPlus small so current is really small

        # if exponent is very negative
        mask = exponent < -100
        gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = (np.abs(exponent) < 100)
        # if exponent is not 0
        mask2 = ((np.abs(exponent) != 0))
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        'compute gammaNegative'
        deltaE = mu_N / e
        exponent = -deltaE / (k_B * T)

        # if exponent is very positive
        mask = exponent > 100
        gammaNegative[mask] = -0.01 # make gammaNegative really small

        # if exponent is very negative
        mask = exponent < -100
        gammaNegative[mask] = (deltaE[mask] / h) * (R_K / R_T[mask]) # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = (np.abs(exponent) < 100)
        # if exponent is not 0
        mask2 = ((np.abs(exponent) != 0))
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaNegative[mask] =  (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        current = e * gammaPlus * gammaNegative / (gammaPlus + gammaNegative)



    else:


        'compute gammaPlus'
        deltaE = -mu_N / e - V_SD
        exponent = deltaE / (k_B * T)

        # if exponent is very positive
        mask = exponent > 100
        gammaPlus[mask] = -0.01  # make gammaPlus small so current is really small

        # if exponent is very negative
        mask = exponent < -100
        gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = (np.abs(exponent) < 100)
        # if exponent is not 0
        mask2 = ((np.abs(exponent) != 0))
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaPlus[mask] = - (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        'compute gammaNegative'

        deltaE = -mu_N / e
        exponent = -deltaE / (k_B * T)

        # if exponent is very positive
        mask = exponent > 100
        gammaNegative[mask] = -0.01  # make gammaNegative really small

        # if exponent is very negative
        mask = exponent < -100
        gammaNegative[mask] = (deltaE[mask] / h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = (np.abs(exponent) < 100)
        # if exponent is not 0
        mask2 = ((np.abs(exponent) != 0))
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaNegative[mask] = (deltaE[mask] / h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        current = -e * gammaPlus * gammaNegative / (gammaPlus + gammaNegative)

    return current

def electricPotential(n, V_SD_grid, V_G_grid, E_C, N_0, e, C_S, C_G):

    """
    Function to compute the electric potential of the QDot.
    :param n: the number of electrons in the dot
    :param V_SD_grid: the 2d array of source-drain voltage values
    :param V_G_grid: the 2d array of gate voltage values
    :return: The Electric Potential for adding the nth electron to the dot
    """

    E_N = E_C*(((n)**2-(n-1)**2)/n*5+random()/9*n)  # arbitrary random formula used to increase diamond width as more electrons are added

    return E_N, (n - N_0 - 1/2) * E_C - (E_C / e) * (C_S * V_SD_grid + C_G * V_G_grid) + E_N





def generate(number_of_examples):
    """
    Function to generate training examples from simulator, calls upon above functions.
    :param number_of_examples: number of simulated examples you wish to generate
    :return: returns True when finished
    """
    # Define Constants

    N_0 = 0
    e = 1.6E-19

    # Define a 1D array for the values for the voltages
    # Define a 1D array for the values for the voltages
    V_SD_max = 0.1
    V_G_min = 0.005
    V_G_max = 1.2
    V_SD = np.linspace(- V_SD_max, V_SD_max, 1000)
    V_G = np.linspace(V_G_min, V_G_max, 1000)

    # Generate 2D array to represent possible voltage combinations

    V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)




    if not os.path.exists("../Training Data/Training_Input"):
        os.makedirs("../Training Data/Training_Input")
    if not os.path.exists("../Training Data/Training_Output"):
        os.makedirs("../Training Data/Training_Output")



    with progressbar.ProgressBar(max_value=number_of_examples) as bar: # initialise progress bar
        for k in range(1,number_of_examples+1):

            seed(datetime.now())  # use current time as random number seed
            N = range(1, randint(1, 15))
            I_tot = np.zeros(V_SD_grid.shape)  # Define the total current
            E_N_previous = 0  # stores previous E_N value
            V_G_start = 0  # start of current diamond
            diamond_starts = np.zeros(
                (1, len(N)))  # numpy array to store the store positions of each diamond along x-axis


            C_S = 10E-19 * uniform(0.1, 1)  # Uniform used for some random variation
            C_D = 10E-19 * uniform(0.2, 1)
            C_G = 1E-18 * uniform(1, 7)
            C = C_S + C_D + C_G
            E_C = (e ** 2) / C

            plt.figure(figsize=(10,10), dpi = 150)

            Estate_height_previous = 0  # stores previous various excited energy height above ground level

            # Charge noise implementation

            alpha = 1E-6 * (C_G / C)

            chargeNoise = np.random.normal(loc=0, scale=alpha, size=V_SD_grid.shape)

            noisy_V_G_grid = V_G_grid + chargeNoise

            for n in N:

                Estate_height = uniform(0.1, 0.5) * E_C
                Lstate_height = uniform(0.5, 0.8) * E_C

                # potential energy of ground to ground transition GS(N-1) -> GS(N)
                E_N, mu_N = electricPotential(n, V_SD_grid, noisy_V_G_grid, E_C, N_0, e, C_S, C_G)
                # Indices where current can flow for  GS(N-1) -> GS(N) transitions
                current_ground = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N, True) + calculate_current(V_SD_grid, V_G_grid, mu_N, False)
                allowed_indices = current_ground != 0

                delta_E_N = E_N - E_N_previous  # Not sure on exact definition yet
                delta_V_G = e/C_G + delta_E_N * C /(e *C_G) # Width of current diamond



                if n == 1:
                    V_G_start = (e / C_G) * (E_N / E_C + 1 / 2)  # start of first diamond / start of current diamond

                    # potential energy of  ground to excited transition GS(N-1) -> ES(N)
                    mu_N_transition1 = mu_N + Estate_height
                    mu_N_transition1 = np.multiply(mu_N_transition1, allowed_indices)
                    current_transition1 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition1,
                                                            True) + calculate_current(V_SD_grid, V_G_grid,
                                                                                      mu_N_transition1, False)
                    random_current_transition1 = current_transition1 * uniform(0.5, 2)

                    I_tot += random_current_transition1 * 0.1

                elif n != 1:
                    V_G_start += delta_V_G  # update so start of current diamond
                    # The transitions from this block are to/from excited states

                    # potential energy of  ground to excited transition GS(N-1) -> ES(N)
                    mu_N_transition1 = mu_N + Estate_height
                    mu_N_transition1 = np.multiply(mu_N_transition1, allowed_indices)
                    current_transition1 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition1, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition1, False)
                    random_current_transition1 = current_transition1 * uniform(0.2, 2)
                    

                    # potential energy of excited to ground transition GS(N-1) -> LS(N)
                    mu_N_transition2 = mu_N + Lstate_height
                    mu_N_transition2 = np.multiply(mu_N_transition2, allowed_indices)
                    current_transition2 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition2, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition2, False)
                    random_current_transition2 = current_transition2 * uniform(0.2, 2)

                    # potential energy of excited to ground transition ES(N-1) -> GS(N)
                    mu_N_transition3 = mu_N - Estate_height_previous
                    mu_N_transition3 = np.multiply(mu_N_transition3, allowed_indices)
                    current_transition3 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition3, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition3, False)
                    random_current_transition3 = current_transition3 * uniform(0.2, 2)

                    # potential energy of excited to ground transition ES(N-1) -> ES(N)
                    mu_N_transition4 = mu_N - Estate_height_previous + Estate_height
                    mu_N_transition4 = np.multiply(mu_N_transition4, allowed_indices)
                    current_transition4 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition4, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition4, False)
                    random_current_transition4 = current_transition4 * uniform(0.2, 2)

                    # potential energy of excited to ground transition ES(N-1) -> LS(N)
                    mu_N_transition5 = mu_N - Estate_height_previous + Lstate_height
                    mu_N_transition5 = np.multiply(mu_N_transition5, allowed_indices)
                    current_transition5 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition5, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition5, False)
                    random_current_transition5 = current_transition5 * uniform(0.2, 2)

                    # potential energy of excited to ground transition LS(N-1) -> GS(N)
                    mu_N_transition6 = mu_N - Lstate_height_previous
                    mu_N_transition6 = np.multiply(mu_N_transition6, allowed_indices)
                    current_transition6 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition6, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition6, False)
                    random_current_transition6 = current_transition6 * uniform(0.2, 2)

                    # potential energy of excited to ground transition LS(N-1) -> ES(N)
                    mu_N_transition7 = mu_N - Lstate_height_previous + Estate_height
                    mu_N_transition7 = np.multiply(mu_N_transition7, allowed_indices)
                    current_transition7 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition7, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition7, False)
                    random_current_transition7 = current_transition7 * uniform(0.2, 2)

                    # potential energy of excited to ground transition LS(N-1) -> LS(N)
                    mu_N_transition8 = mu_N - Lstate_height_previous + Lstate_height
                    mu_N_transition8 = np.multiply(mu_N_transition8, allowed_indices)
                    current_transition8 = calculate_current(V_SD_grid, noisy_V_G_grid, mu_N_transition8, True) + calculate_current(V_SD_grid, V_G_grid, mu_N_transition8, False)
                    random_current_transition8 = current_transition8 * uniform(0.2, 2)

                    I_tot += (random_current_transition1 + random_current_transition2 + random_current_transition3 + \
                              random_current_transition4 + random_current_transition5 + random_current_transition6 + \
                              random_current_transition7 + random_current_transition8) * 0.1

                diamond_starts[0, n - 1] = V_G_start


                I_tot += current_ground
                # update 'previous' variables to previous values
                E_N_previous = E_N
                Estate_height_previous = Estate_height
                Lstate_height_previous = Lstate_height

            thermaNoise = np.random.normal(loc=0, scale=1, size=V_SD_grid.shape)
            k_B = 8.6173E-5 * e
            g_0 = I_tot / noisy_V_G_grid
            T = 0.01
            I_thermalNoise = np.sqrt(4 * k_B * T * np.abs(g_0)) * thermaNoise
            I_tot += I_thermalNoise

            shotNoise = np.random.normal(loc=0, scale=1, size=V_SD_grid.shape)
            delta_f = 1000  # bandwidth
            I_shotNoise = np.sqrt(2 * e * np.abs(I_tot) * delta_f) * shotNoise
            I_tot += I_shotNoise

            #I_max = np.max(I_tot)
            #I_min = np.min(I_tot)
            #print("Maximum amount of thermal noise is", np.max(I_thermalNoise))
            #print("Maximum amount of shot noise is", np.max(I_shotNoise))
            I_tot_abs = np.abs(I_tot)
            I_max_abs = np.max(I_tot_abs)
            I_min_abs = np.min(I_tot_abs)

            # Plot diamonds
            contour = plt.contourf(V_G_grid,V_SD_grid, I_tot_abs, cmap="seismic", levels = np.linspace(I_min_abs,I_max_abs,500)) # draw contours of diamonds

            #contour = plt.contourf(V_G_grid,V_SD_grid, I_tot, cmap="seismic", levels = np.linspace(I_min,I_max,500)) # draw contours of diamonds
            '''The extra diamonds arose out of the fact that there was a small number of contour levels added in 
            levels attribute to fix this so 0 current was grouped with the small current values '''
            plt.ylim([-V_SD_max, V_SD_max])
            plt.xlim([V_G_min, V_G_max])
            #plt.xlabel("$V_{G}$ / V")
            #plt.ylabel("$V_{SD}$ / V")

            #plt.title("Single Quantum Dot Coulomb Blockade Simulation")
            #cbar = plt.colorbar(contour)
            #cbar.set_label("$I$ / A")

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(plt.NullLocator()) # trick found on stackex. when trying to get rid of padding
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.savefig("../Training Data/Training_Input/input_{0}.png".format(k), bbox_inches='tight', pad_inches=0.0)
            plt.close()

            # Compute negative and positive slopes of diamonds for drawing edges
            positive_slope = C_G / (C_G + C_D)
            negative_slope = - C_G / C_S

            plt.figure(figsize=(10, 10), dpi=150)

            for i in range(
                    len(N) - 1):  # need -1 as block would attempt to access index N otherwise and it doesn't exist
                # positive grad. top-left
                x_final = (positive_slope * diamond_starts[0, i] - negative_slope * diamond_starts[0, i + 1]) / (
                            positive_slope - negative_slope)  # analytical formula derived by equating equations of lines
                x_values = [diamond_starts[0, i], x_final]
                y_final = positive_slope * (x_final - diamond_starts[0, i])
                y_values = [0, y_final]
                plt.plot(x_values, y_values, '-k')

                # negative grad. top-right
                x_values = [x_final, diamond_starts[0, i + 1]]
                y_values = [y_final, 0]
                plt.plot(x_values, y_values, '-k')

                # positive grad. bottom-right
                x_final = (positive_slope * diamond_starts[0, i + 1] - negative_slope * diamond_starts[0, i]) / (
                            positive_slope - negative_slope)
                x_values = [diamond_starts[0, i + 1], x_final]
                y_final = positive_slope * (x_final - diamond_starts[0, i + 1])
                y_values = [0, y_final]
                plt.plot(x_values, y_values, '-k')

                # negative grad. bottom-left
                x_values = [x_final, diamond_starts[0, i]]
                y_values = [y_final, 0]
                plt.plot(x_values, y_values, '-k')

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # trick found on stackex. when trying to get rid of padding
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.ylim([-V_SD_max, V_SD_max])
            plt.xlim([V_G_min, V_G_max])

            plt.xlabel("$V_{G}$ / V")
            plt.ylabel("$V_{SD}$ / V")

            plt.savefig("../Training Data/Training_Output/output_{0}.png".format(k), bbox_inches='tight', pad_inches=0.0)
            plt.close()




            bar.update(k) # update progress bar
    return True






