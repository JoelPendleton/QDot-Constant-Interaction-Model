# -----------------------------------------------------------
# Single Quantum Dot Simulator that is used to Generate Training Examples for a CNN.
#
# (C) 2021 Joel Pendleton, London, UK
# Released under MIT license
# email joel.pendleton@quantummotion.tech
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import seed # generates seed for random number generator
from random import random  # random generates a random number between 0 and 1
from random import uniform # generates random float between specified range
from random import randint
from datetime import datetime
import xml.etree.cElementTree as ET
import warnings
warnings.filterwarnings("ignore")

class QuantumDot:

    """
    This is a class for the simulation of a single quantum dot system using the constant interaction model.

    Attributes:
        simCount (int): The number of instances of this class
        e (float): electron's charge. Units: C
        h (float): Planck's constant. Units: eVs
        T (float): temperature of system. Units: K
        k_B (float): Boltzmann's constant. Units: eVK^-1
        V_SD_max(float): range of source-drain voltage values. Units: V
        V_G_min (float): minimum value of gate voltage. Units: V
        V_G_max (float): maximum value of gate voltage. Units: V
        V_SD(float): 1D numpy array of <volt_steps> source-drain voltage values. Units: V
        V_G(float): 1D numpy array of <volt_steps> gate voltage values. Units: V
        V_SD_grid (float): 2D numpy array of <volt_steps> x <volt_steps> source-drain voltage values. Units: V
        V_G_grid (float): 2D numpy array of <volt_steps> x <volt_steps> gate voltage values. Units: V
    """
    seed(datetime.now())
    simCount = 0 # number of simulations
    e = 1.6E-19
    h = 4.1357E-15
    k_B = 8.6173E-5
    T = 0.01

    # These following voltage parameters are chosen to prevent the following issue
    # Previously, I had an issue where when an image of the simulation was generated the bounding boxes would be skewed
    # The reasons for this were
        # The number of steps/voltage values for the individual y and x axis didn't match
        # The difference between each of the voltage values for the axis (the steps) didn't match
    # This mean't when trying to output a 500/500 square the rectangular bounding boxes were skewed.

    V_SD_max = 0.2
    V_G_min = 0.02
    V_G_max = 0.42
    step_size = 0.005

    DPI = 100
    image_hw = 500 # pixel resolution of image (image_hw x image_hw)
    V_SD_offset = uniform(0.00001, 0.0001)
    V_G_offset = uniform(0.00001, 0.0001)

    # We need this otherwise the plot axes get shifted and there is whitespace
    V_G_max -= V_G_offset
    V_G_min -= V_G_offset
    V_SD_max -= V_SD_offset


    V_SD = np.linspace(- V_SD_max, V_SD_max, 100) + V_SD_offset# random factor added so the diamonds
    # V_SD = np.arange(- V_SD_max, V_SD_max, step_size) + V_SD_offset# random factor added so the diamonds
    # aren't always in the vertical centre.

    V_G = np.linspace(V_G_min, V_G_max, 100) + V_G_offset
    # V_G = np.arange(V_G_min, V_G_max, step_size) #+ V_G_offset

    # Generate 2D array to represent possible voltage combinations
    V_SD_grid, V_G_grid = np.meshgrid(V_SD, V_G)


    mu_S = - e * V_SD_grid  # source electrochemical potential energy

    def __init__(self):
        """
       The constructor for QuantumDot class.

       Parameters:
          N (int): list containing the numbers of the diamonds drawn / electrons to be added to the system.
          N_0 (int):
          I_tot (float): 2d numpy array containing the total current matrix of the system. Units: A
          diamond_starts (float): 1d numpy array containing start positions of diamonds along x-axis when V_SD = 0. Units: V
          C_S (float): source capacitance. Units: F
          C_D(float): drain capacitance. Units: F
          C_G (float): gate capacitance. Units: F
          C (float): total system capacitance. Units: F
          E_C (float): charging energy. Units: eV

       """
        QuantumDot.simCount += 1
        seed(datetime.now())  # use current time as random number seed
        self.N = range(1, 25)
        self.N_0 = 0 # I think making this non-zero offsets the bounding boxes used for object detection
        self.I_tot = np.zeros(self.V_SD_grid.shape)
        self.diamond_starts = np.zeros((1, len(self.N)))
        self.C_S = 10E-19 * uniform(0.1, 2)  # Uniform used for some random variation
        self.C_D = 10E-19 * uniform(0.2, 2)
        self.C_G = 1E-18 * uniform(1, 9)
        self.C = self.C_S + self.C_D + self.C_G
        self.E_C = (self.e ** 2) / self.C


    def electricPotential(self, n):

        """
        The function to compute electrochemical potential of current system.

        Parameters:
            n (int): number of electrons currently in the system / quantum dot.

        Returns:
            E_N (float): the chemical potential energy of the system.
            mu_N (float): the electrochemical potential of the system
        """

        # arbitrary formula used to increase diamond width as more electrons are added
        E_N = self.E_C * (n ** 2 - (n - 1) ** 2)# / n * 5 #+ random() / 9 * n)

        mu_N = (n - self.N_0 - 1 / 2) * self.E_C - (self.E_C / self.e) * (self.C_S * self.V_SD_grid + self.C_G * self.V_G_grid) + E_N

        return E_N, mu_N

    def currentChecker(self, mu_N):

        """
        Function to determne region where current is allowed to flow and where there is a blockade.
        Finds indexes corresponding to values of V_SD and V_G for which current can flow from source-drain or drain-source.

        Parameters:
            mu_N (float): the electrochemical potential of the transition.

        Returns:
            Matrix of booleans. True corresponds to position of allowed current flow, False corresponds to no current flow.
        """
        # the algorithm below looks contrived but it removes the need for for loops increasing runtime
        # it checks whether the potential energy of the electron state is between the source and drain
        condition1 = mu_N > 0
        condition2 = mu_N < self.mu_S
        condition3 = self.V_SD_grid < 0
        condition4 = mu_N < 0
        condition5 = mu_N > self.mu_S
        condition6 = self.V_SD_grid > 0
        # Consider both scenarios where mu_D < mu_N < mu_S and mu_S < mu_N < mu_D
        I_1 = (condition1 & condition2 & condition3)
        I_2 = (condition4 & condition5 & condition6)
        return np.logical_or(I_1, I_2)  # combine the result of these possibilities.

    def calculate_current(self, V_SD, V_G, mu_N):

        """
        The function to compute the current of the specified transition.

        Parameters:
            mu_N (float): the electrochemical potential of the transition.

        Returns:
            current (float): the current associated with the transition.
        """

        R_K = self.h / (self.e ** 2)
        R_T = 1 / (1E-26 * V_G ** 2)

        gammaPlus = np.ones(V_G.shape)
        gammaNegative = np.ones(V_G.shape)
        current = np.zeros(V_G.shape)

        'compute gammaPlus for source -> drain current'
        deltaE = mu_N / self.e + V_SD
        exponent = deltaE / (self.k_B * self.T)

        # if exponent is very positive
        mask = exponent > 100
        gammaPlus[mask] = -0.01  # make gammaPlus small so current is really small

        # if exponent is very negative
        mask = exponent < -100
        gammaPlus[mask] = - (deltaE[mask] / self.h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = np.abs(exponent) < 100
        # if exponent is not 0
        mask2 = np.abs(exponent) != 0
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaPlus[mask] = - (deltaE[mask] / self.h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        'compute gammaNegative for source -> drain current'
        deltaE = mu_N / self.e
        exponent = -deltaE / (self.k_B * self.T)

        # if exponent is very positive
        mask = exponent > 100
        gammaNegative[mask] = -0.01  # make gammaNegative really small

        # if exponent is very negative
        mask = exponent < -100
        gammaNegative[mask] = (deltaE[mask] / self.h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = np.abs(exponent) < 100
        # if exponent is not 0
        mask2 = np.abs(exponent) != 0
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaNegative[mask] = (deltaE[mask] / self.h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        current += self.e * gammaPlus * gammaNegative / (gammaPlus + gammaNegative)

        'compute gammaPlus for drain -> source current'
        deltaE = -mu_N / self.e - V_SD
        exponent = deltaE / (self.k_B * self.T)

        # if exponent is very positive
        mask = exponent > 100
        gammaPlus[mask] = -0.01  # make gammaPlus small so current is really small

        # if exponent is very negative
        mask = exponent < -100
        gammaPlus[mask] = - (deltaE[mask] / self.h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = np.abs(exponent) < 100
        # if exponent is not 0
        mask2 = np.abs(exponent) != 0
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaPlus[mask] = - (deltaE[mask] / self.h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        'compute gammaNegative for drain -> source current'

        deltaE = -mu_N / self.e
        exponent = -deltaE / (self.k_B * self.T)

        # if exponent is very positive
        mask = exponent > 100
        gammaNegative[mask] = -0.01  # make gammaNegative really small

        # if exponent is very negative
        mask = exponent < -100
        gammaNegative[mask] = (deltaE[mask] / self.h) * (R_K / R_T[mask])  # disregard exponential term

        # if exponent is between -100 and 100
        mask1 = np.abs(exponent) < 100
        # if exponent is not 0
        mask2 = np.abs(exponent) != 0
        # if exponent is between -100 and 100, and not equal to 0
        mask = np.logical_and(mask1, mask2)
        # perform full calculation
        gammaNegative[mask] = (deltaE[mask] / self.h) * (R_K / R_T[mask]) / (1 - np.exp(exponent[mask]))

        current += -self.e * gammaPlus * gammaNegative / (gammaPlus + gammaNegative)

        return current


    def simulate(self, simulation_number, noise, path):

        """
        The function to simulate the system, and produce a contour plot of current against source-drain and gate voltage
        as well as a plot of the edges.

        Contour plot saved to "./data/train/image/"
        Bounding box parameters saved to "./data/train/annotation/"

        Parameters:
            simulation_number (int): the number of this simulation / file name suffix.
            noise (boolean): true if noise is to be added to simulation, and false if not.
            path (str): location to
        Returns:
            True if simulation was successful
            False if no diamonds are within the voltage space (not successful)
        """

        fig = plt.figure(simulation_number, figsize=(self.image_hw/self.DPI,self.image_hw/self.DPI))
        ax = fig.add_axes([0, 0, 1, 1])


        E_N_previous = 0
        Estate_height_previous = 0  # stores previous various excited energy height above ground level
        V_G_start = 0  # start of current diamond
        allowed_indices = np.zeros(self.V_SD_grid.shape)
        I_ground = np.zeros(self.V_SD_grid.shape)
        I_excited = np.zeros(self.V_SD_grid.shape)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

        for n in self.N:

            if noise:
                'Charge noise implementation'
                alpha = 1E-6 * (self.C_G / self.C)
                chargeNoise = np.random.normal(loc=0, scale=alpha, size=self.V_SD_grid.shape)
                noisy_V_G_grid = self.V_G_grid + chargeNoise
            else:

                noisy_V_G_grid = self.V_G_grid

            Estate_height = uniform(0.1, 0.3) * self.E_C
            Lstate_height = uniform(0.3, 0.5) * self.E_C

            # potential energy of ground to ground transition GS(N-1) -> GS(N)
            E_N, mu_N = self.electricPotential(n)
            # Indices where current can flow for  GS(N-1) -> GS(N) transitions
            I_ground += self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N)
            allowed_indices = np.logical_or(allowed_indices, self.currentChecker(mu_N))

            delta_E_N = E_N - E_N_previous
            delta_V_G = self.e / self.C_G + delta_E_N * self.C / (self.e * self.C_G)  # Width of current diamond

            if n == 1:
                V_G_start = (self.e / self.C_G) * (E_N / self.E_C + 1 / 2)  # start of first diamond / start of current diamond

                # potential energy of  ground to excited transition GS(N-1) -> ES(N)
                mu_N_transition1 = mu_N + Estate_height
                current_transition1 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition1)

                I_excited += current_transition1

            elif n != 1:
                V_G_start += delta_V_G  # update so start of current diamond
                # The transitions from this block are to/from excited states

                # potential energy of ground to excited transition GS(N-1) -> ES(N)
                mu_N_transition1 = mu_N + Estate_height
                current_transition1 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition1)

                # potential energy of excited to ground transition GS(N-1) -> LS(N)
                mu_N_transition2 = mu_N + Lstate_height
                current_transition2 =self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition2)

                # potential energy of excited to ground transition ES(N-1) -> GS(N)
                mu_N_transition3 = mu_N - Estate_height_previous
                current_transition3 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition3)

                # potential energy of excited to ground transition ES(N-1) -> ES(N)
                mu_N_transition4 = mu_N - Estate_height_previous + Estate_height
                current_transition4 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition4)

                # potential energy of excited to ground transition ES(N-1) -> LS(N)
                mu_N_transition5 = mu_N - Estate_height_previous + Lstate_height
                current_transition5 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition5)

                # potential energy of excited to ground transition LS(N-1) -> GS(N)
                mu_N_transition6 = mu_N - Lstate_height_previous
                current_transition6 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition6)

                # potential energy of excited to ground transition LS(N-1) -> ES(N)
                mu_N_transition7 = mu_N - Lstate_height_previous + Estate_height
                current_transition7 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition7)

                # potential energy of excited to ground transition LS(N-1) -> LS(N)
                mu_N_transition8 = mu_N - Lstate_height_previous + Lstate_height
                current_transition8 = self.calculate_current(self.V_SD_grid, noisy_V_G_grid, mu_N_transition8)

                I_excited += current_transition1 + current_transition2 + current_transition3 +\
                                        current_transition4 + current_transition5 + current_transition6 + \
                                        current_transition7 + current_transition8

            self.diamond_starts[0, n - 1] = V_G_start

            # Update 'previous' variables to previous values
            E_N_previous = E_N
            Estate_height_previous = Estate_height
            Lstate_height_previous = Lstate_height

        self.I_tot += I_ground + np.multiply(I_excited, allowed_indices)

        if noise:
            'Thermal noise implementation'
            thermalNoise = np.random.normal(loc=0, scale=1, size=self.V_SD_grid.shape)
            g_0 = self.I_tot / self.V_SD_grid
            I_thermalNoise = np.sqrt(4 * self.k_B * self.e * self.T * np.abs(g_0)) * thermalNoise
            self.I_tot += I_thermalNoise # when I uncomment this makes the image white in some cases as it is much larger than the normal current

            'SHOT noise implementation'
            shotNoise = np.random.normal(loc=0, scale=1, size=self.V_SD_grid.shape)
            delta_f = 1000  # bandwidth
            I_shotNoise = np.sqrt(2 * self.e * np.abs(self.I_tot) * delta_f) * shotNoise
            self.I_tot += I_shotNoise

        # Ensure no current flows inside diamonds from excited states since it's forbidden
        self.I_tot = np.multiply(allowed_indices, self.I_tot)
        I_tot_abs = np.abs(self.I_tot)

        offset = np.min(I_tot_abs)
        self.I_tot += np.multiply(np.logical_not(allowed_indices), offset)


        I_max_abs = np.max(I_tot_abs)
        I_min_abs = np.min(I_tot_abs)
        I_grad_V_SD, I_grad_V_G = np.gradient(I_tot_abs)

        # Plot diamonds (current)
        ax.contourf(self.V_G_grid, self.V_SD_grid, I_tot_abs, cmap="seismic",
                               levels=np.linspace(I_min_abs, I_max_abs, 500))  # draw contours of diamonds

        ax.axis('off')

        # Plot diamonds (gradient)
        # plt.contourf(self.V_G_grid, self.V_SD_grid, np.abs(I_grad_V_SD), cmap="gray",
        #              levels=np.linspace(np.min(I_grad_V_SD), np.max(I_grad_V_SD), 500))  # draw contours of diamonds


        plt.ylim([-self.V_SD_max, self.V_SD_max])
        plt.xlim([self.V_G_min, self.V_G_max])

        'Generate Training Annotation'

        # Compute negative and positive slopes of diamonds for drawing edges
        positive_slope = self.C_G / (self.C_G + self.C_D)
        negative_slope = - self.C_G / self.C_S

        diamonds_visible = 0

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "path").text = str(path + "images/{0}.png".format(simulation_number))

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Quantum Dot Simulator"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(self.image_hw)
        ET.SubElement(size, "height").text = str(self.image_hw)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(root, "segmented").text = "0"


        for i in range(
                len(self.N) - 1):  # need -1 as block would attempt to access index N otherwise and it doesn't exist

            # Parameters in standard units / true values
            width = self.diamond_starts[0, i + 1] - self.diamond_starts[0, i]
            D_x = self.diamond_starts[0, i] # start of diamond
            D_y = self.V_SD_offset
            B_x = self.diamond_starts[0, i] + width # end of diamond
            B_y = self.V_SD_offset
            # positive grad. top-left
            A_x = (positive_slope * self.diamond_starts[0, i] - negative_slope * self.diamond_starts[0, i + 1]) / (
                   positive_slope - negative_slope)  # analytical formula derived by equating equations of lines
            A_y = positive_slope * (A_x - self.diamond_starts[0, i])

            # positive grad. bottom-right
            C_x = (positive_slope * self.diamond_starts[0, i + 1] - negative_slope * self.diamond_starts[0, i]) / (
                    positive_slope - negative_slope)
            C_y = positive_slope * (C_x - self.diamond_starts[0, i + 1])

            # Compute relevant variables for the computation of bounding box coordinates
            theta = np.arctan(positive_slope)
            alpha = np.abs((np.arctan(negative_slope)))
            BtoA = np.sqrt((B_x - A_x)**2 + (B_y - A_y)**2)
            DtoC = np.sqrt((C_x - D_x)**2 + (C_y - D_y)**2)
            BtoP = BtoA * np.cos(np.pi - alpha - theta)
            DtoQ = DtoC * np.cos(np.pi - alpha - theta)
            P_x = B_x + BtoP * np.cos(theta)
            P_y = B_y + BtoP * np.sin(theta)
            Q_x = D_x - DtoQ * np.cos(theta)
            Q_y = D_y - DtoQ * np.sin(theta)


            # turn true values of bounding box coordinates into pixel coordinates
            A_x_scaled, A_y_scaled = ax.transData.transform((A_x,A_y))
            A_y_scaled = self.image_hw - A_y_scaled
            C_x_scaled, C_y_scaled = ax.transData.transform((C_x,C_y))
            C_y_scaled = self.image_hw - C_y_scaled
            Q_x_scaled, Q_y_scaled = ax.transData.transform((Q_x,Q_y))
            Q_y_scaled = self.image_hw - Q_y_scaled
            P_x_scaled, P_y_scaled = ax.transData.transform((P_x,P_y))
            P_y_scaled = self.image_hw - P_y_scaled


            # ensure bounding boxes within 10px from edge are not detected/noted
            condition_1 = (P_x_scaled < (self.image_hw)) and (Q_x_scaled > 0)
            condition_2 = (C_y_scaled < (self.image_hw)) and (A_y_scaled > 0)
            if (condition_1 and condition_2):

                xy = np.array([[A_x,A_y],
                              [P_x,P_y],
                              [C_x,C_y],
                              [Q_x,Q_y]])
                # rect = patches.Polygon(xy, linewidth=1, edgecolor='g', facecolor='none')
                # ax.add_patch(rect)


                object = ET.SubElement(root, "object")
                ET.SubElement(object, "name").text = "diamond"
                ET.SubElement(object, "pose").text = "Unspecified"
                ET.SubElement(object, "truncated").text = "0"
                ET.SubElement(object, "difficult").text = "0"

                bndbox = ET.SubElement(object, "bndbox")


                ET.SubElement(bndbox, "x0").text = str(A_x_scaled)
                ET.SubElement(bndbox, "y0").text = str(A_y_scaled)

                ET.SubElement(bndbox, "x1").text = str(P_x_scaled)
                ET.SubElement(bndbox, "y1").text = str(P_y_scaled)

                ET.SubElement(bndbox, "x2").text = str(C_x_scaled)
                ET.SubElement(bndbox, "y2").text = str(C_y_scaled)

                ET.SubElement(bndbox, "x3").text = str(Q_x_scaled)
                ET.SubElement(bndbox, "y3").text = str(Q_y_scaled)

                diamonds_visible += 1

            else:
                continue

        if diamonds_visible < 1:
            #print("Retrying simulation of Quantum Dot", simulation_number)
            return False
        else:
            fig.savefig(path + "images/{0}.png".format(simulation_number), dpi=(self.DPI))  # Save training image
            tree = ET.ElementTree(root)
            tree.write(path + "labeltxt/{0}.xml".format(simulation_number))
            plt.close(simulation_number)
            return True

