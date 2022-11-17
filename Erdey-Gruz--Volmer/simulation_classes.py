import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import exp_approximations as ea

F = 96485
z = 1
R = 8.3145
T = 298.15

class Cell:
    def __init__(self, x: float, dx: float, cinf: float, D: float, A: float, E0: float, Rsol:float, k: float, alpha: float, name: str):
        self.x = x
        self.dx = dx
        self.cinf = cinf
        self.D = D
        self.A = A
        self.E0 = E0
        self.Rsol = Rsol
        self.k = k
        self.alpha = alpha
        self.name = name
        self.createCell()
        self.current = []

    def __str__(self):
        return str(self.cell)

    def createCell(self):
        self.numOfElements = int(self.x / self.dx)
        self.cell = [np.full(self.numOfElements, self.cinf) for i in range(2)]

    def electrodeReaction(self, setVolt: float, dt: float):
        global F
        global z
        global R
        global T
        if self.Rsol == 0:
            kOx = (self.k / self.dx) * np.exp(self.alpha * z * F * (setVolt - self.E0) / (R * T))
            kRed = (self.k / self.dx) * np.exp(-1*(1-self.alpha) * z * F * (setVolt - self.E0) / (R * T))
            cTot = self.cell[0][0] + self.cell[1][0]
            newOx = ( (kOx * cTot) + (kRed * self.cell[0][0] - kOx * self.cell[1][0])*np.exp( -1 * (kOx + kRed) * dt ) ) / (kOx + kRed)
            newRed = ( (kRed * cTot) + (kOx * self.cell[1][0] - kRed * self.cell[0][0])*np.exp( -1 * (kOx + kRed) * dt ) ) / (kOx + kRed)
            newCurrent = z * F * self.A * self.dx * (newOx - self.cell[0][0]) / self.dt
        else:
            epsilon = 1e-4
            def difference(effectiveVolt: float):
                kOx = (self.k / self.dx) * np.exp(self.alpha * z * F * (effectiveVolt - self.E0) / (R * T))
                kRed = (self.k / self.dx) * np.exp(-1*(1-self.alpha) * z * F * (setVolt - self.E0) / (R * T))
                cTot = self.cell[0][0] + self.cell[1][0]
                newOx = ( (kOx * cTot) + (kRed * self.cell[0][0] - kOx * self.cell[1][0])*np.exp( -1 * (kOx + kRed) * dt ) ) / (kOx + kRed)
                newCurrent = z * F * self.A * self.dx * (newOx - self.cell[0][0]) / self.dt
                return setVolt - (effectiveVolt + newCurrent * self.Rsol)
            #sol = sp.optimize.root_scalar(difference, method = 'toms748', bracket = [-10, 10], x0 = setVolt, rtol = epsilon)
            sol = sp.optimize.root_scalar(difference, method = 'secant', x0 = setVolt-0.1, x1 = setVolt+0.1, rtol = epsilon)
            effectiveVolt = sol.root
            kOx = (self.k / self.dx) * np.exp(self.alpha * z * F * (effectiveVolt - self.E0) / (R * T))
            kRed = (self.k / self.dx) * np.exp(-1*(1-self.alpha) * z * F * (effectiveVolt - self.E0) / (R * T))
            cTot = self.cell[0][0] + self.cell[1][0]
            newOx = ( (kOx * cTot) + (kRed * self.cell[0][0] - kOx * self.cell[1][0])*np.exp( -1 * (kOx + kRed) * dt ) ) / (kOx + kRed)
            newRed = ( (kRed * cTot) + (kOx * self.cell[1][0] - kRed * self.cell[0][0])*np.exp( -1 * (kOx + kRed) * dt ) ) / (kOx + kRed)
            newCurrent = z * F * self.A * self.dx * (newOx - self.cell[0][0]) / self.dt
        self.current.append(newCurrent)
        self.cell[0][0] = newOx
        self.cell[1][0] = newRed

    def setdt(self, dt: float, padeparams: tuple):
        self.dt = dt
        transformMatrix = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape = (self.numOfElements, self.numOfElements))
        transformMatrix = transformMatrix.tocsc()
        transformMatrix[0, 0] = -1
        transformMatrix[self.numOfElements-1, self.numOfElements-1] = -1
        transformMatrix = self.D * dt / (self.dx**2) * transformMatrix
        if padeparams[0] == -1:
            transformMatrix = sp.sparse.linalg.expm(transformMatrix)
        else:
            transformMatrix = ea.general_pade(padeparams[0], padeparams[1], transformMatrix)
        self.__transformMatrix = transformMatrix

    def propagate(self):
        for i in range(2):
            self.cell[i] = self.__transformMatrix.dot(self.cell[i])

    def modifyParameter(self, param: str, newVal: str | float):
        setattr(self, param, newVal)
        self.createCell()

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, dx={self.dx}, cinf={self.cinf}, D={self.D})"


class Experiment:
    def __init__(self, t: float, dt: float, Vmin: float, Vmax: float, sweepRate: float, startVoltage: float, name: str):
        self.t = t
        self.dt = dt
        self.name = name
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.sweepRate = sweepRate
        self.startVoltage = startVoltage
        self.numOfTimesteps = int(self.t / self.dt)
        self.cells = []
        self.cellnames = []
        self.calculateVoltages()

    def calculateVoltages(self):
        tcoords = np.linspace(0, self.t, int(self.t/self.dt), False)
        amplitude = (self.Vmax - self.Vmin) / 2
        offset = (self.Vmax + self.Vmin) / 2
        period = amplitude / self.sweepRate
        self.voltages = ( ( 4 * amplitude / period ) * abs( ( ( tcoords  - ( (self.startVoltage - self.Vmax) * period / ((self.Vmin - self.Vmax) * 2) )) % period) - ( period / 2 ) ) ) - amplitude + offset

    def addCell(self, x: float, dx: float, cinf: float, D: float, A: float, E0: float, Rsol: float, k: float, alpha: float, name: str = "", padeparams: tuple = (-1,)):
        if not name:
            name = str(len(self.cells))
        newCell = Cell(x, dx, cinf, D, A, E0, Rsol, k, alpha, name)
        newCell.setdt(self.dt, padeparams)
        self.cells.append(newCell)
        self.cellnames.append(name)

    def modifyCellParameter(self, cellID: str | int, param: str, newVal: str | float, padeparams: tuple):
        if type(cellID) == int:
            index = cellID
        else:
            index = self.cellnames.index(cellID)
        self.cells[index].modifyParameter(param, newVal)
        if param == "name":
            self.cellnames[index] = newVal
        else:
            self.cells[index].setdt(self.dt, padeparams)

    def modifyExperimentParameter(self, param: str, newVal: str | float, padeparams: tuple):
        setattr(self, param, newVal)
        if param == "t":
            self.numOfTimesteps = int(self.t / self.dt)
        elif param == "dt":
            self.numOfTimesteps = int(self.t / self.dt)
            for cell in self.cells:
                cell.setdt(self.dt, padeparams)
    
    def electrodeReaction(self, voltage: float):
        for cell in self.cells:
            cell.electrodeReaction(voltage, self.dt)

    def propagate(self):
        for cell in self.cells:
            cell.propagate()

    def simulate(self):
        for voltage in self.voltages:
            self.electrodeReaction(voltage)
            self.propagate()

    def __str__(self):
        if not self.cells:
            experimentString = "No cells in the experiment."
            return experimentString
        experimentString = "Cells in this experiment:\n"
        for cell in self.cells:
            experimentString += "Cell " + cell.name + ": " + str(cell.cell) + "\n"
        return experimentString

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return f"{self.__class__.__name__} \"{self.name}\" (t={self.t}, dt={self.dt}, number of cells={len(self.cells)})"

    def plot(self):
        for cell in self.cells:
            xcoords = np.linspace(0, cell.x, cell.numOfElements, False)
            plt.plot(xcoords, cell.cell[0], label = cell.name+" Ox")
            plt.plot(xcoords, cell.cell[1], label = cell.name+" Red")
        plt.legend()
        plt.show()

    def plotNoShow(self, plot: plt.figure):
        for cell in self.cells:
            xcoords = np.linspace(0, cell.x, cell.numOfElements, False)
            plt.plot(xcoords, cell.cell[0], label = cell.name+" Ox")
            plt.plot(xcoords, cell.cell[1], label = cell.name+" Red")
        plot.legend()

    def plotCV(self):
        for cell in self.cells:
            plt.plot(self.voltages, cell.current, label = cell.name)
        plt.legend()
        plt.show()
    
    def plotCVNoShow(self, plot: plt.figure):
        for cell in self.cells:
            plt.plot(self.voltages, cell.current, label = cell.name)
        plt.legend()


class Simulation:
    def __init__(self, name: str, padeparams: tuple = (-1,)):
        self.padeparams = padeparams
        self.name = name
        self.experiments = []
        self.experimentnames = []
    
    def addExperiment(self, t: float, dt: float, Vmin: float, Vmax: float, sweepRate: float, startVoltage: float, name: str = ""):
        if not name:
            name = str(len(self.experiments))
        newExperiment = Experiment(t, dt, Vmin, Vmax, sweepRate, startVoltage, name)
        self.experiments.append(newExperiment)
        self.experimentnames.append(name)

    def addCell(self, experimentID: str | int, x: float, dx: float, cinf: float, D: float, A: float, E0: float, Rsol: float, k: float, alpha: float, name: str = ""):
        if type(experimentID) == str:
            index = self.experimentnames.index(experimentID)
        else:
            index = experimentID
        if experimentID != -1:
            self.experiments[index].addCell(x, dx, cinf, D, A, E0, Rsol, k, alpha, name, self.padeparams)
        else:
            for experiment in self.experiments:
                experiment.addCell(x, dx, cinf, D, A, E0, Rsol, k, alpha, name, self.padeparams)

    def modifySimulationParameter(self, param: str, newVal: tuple | str):
        setattr(self, param, newVal)
        for experiment in self.experiments:
            experiment.modifyExperimentParameter("dt", experiment.dt, self.padeparams)

    def modifyExperimentParameter(self, experimentID: str | int, param: str, newVal: float):
        if type(experimentID) == int:
            index = experimentID
        else:
            index = self.experimentnames.index(experimentID)
        self.experiments[index].modifyExperimentParameter(param, newVal, self.padeparams)
        if param == "name":
            self.experimentnames[index] = newVal

    def modifyCellParameter(self, experimentID: str | int, cellID: str | int, param: str, newVal: float):
        if type(experimentID) == int:
            index = experimentID
        else:
            index = self.experimentnames.index(experimentID)
        self.experiments[index].modifyCellParameter(cellID, param, newVal, self.padeparams)

    def simulate(self):
        for experiment in self.experiments:
            experiment.simulate()

    def __str__(self):
        if not self.experiments:
            simulationString = "No experiments in the simulation."
            return simulationString
        simulationString = "Experiments in this simulation:\n"
        for experiment in self.experiments:
            simulationString += "Experiment " + experiment.name + "\n"
        return simulationString

    def __len__(self):
        return len(self.experiments)

    def __repr__(self):
        return f"{self.__class__.__name__}(number of experiments={len(self.experiments)})"

    def plot(self):
        for experiment in self.experiments:
            figure = plt.figure()
            figure.suptitle('Simulation ' + self.name + ': ' + repr(experiment))
            subplot = figure.add_subplot()
            experiment.plotNoShow(subplot)
        plt.show()

    def plotNoShow(self):
        for experiment in self.experiments:
            figure = plt.figure()
            figure.suptitle('Simulation ' + self.name + ': ' + repr(experiment))
            subplot = figure.add_subplot()
            experiment.plotNoShow(subplot)

    def plotCV(self):
        for experiment in self.experiments:
            figure = plt.figure()
            figure.suptitle('Simulation ' + self.name + ': ' + repr(experiment))
            subplot = figure.add_subplot()
            experiment.plotCVNoShow(subplot)
        plt.show()

    def plotNoShow(self):
        for experiment in self.experiments:
            figure = plt.figure()
            figure.suptitle('Simulation ' + self.name + ': ' + repr(experiment))
            subplot = figure.add_subplot()
            experiment.plotCVNoShow(subplot)

        

class Program:
    def __init__(self):
        self.simulations = []
        self.simulationnames = []
    
    def addSimulation(self, name: str, padeparams: tuple = (-1,)):
        if not name:
            name = str(len(self.simulations))
        newSimulation = Simulation(name, padeparams)
        self.simulations.append(newSimulation)
        self.simulationnames.append(name)

    def addExperiment(self, simulationID: str | int, t: float, dt: float, Vmin: float, Vmax: float, sweepRate: float, startVoltage: float, name: str = ""):
        if type(simulationID) == str:
            index = self.simulationnames.index(simulationID)
        else:
            index = simulationID
        if simulationID != -1:
            self.simulations[index].addExperiment(t, dt, Vmin, Vmax, sweepRate, startVoltage, name)
        else:
            for simulation in self.simulations:
                simulation.addExperiment(t, dt, Vmin, Vmax, sweepRate, startVoltage, name)

    def addCell(self, simulationID: str | int, experimentID: str | int, x: float, dx: float, cinf: float, D: float, A: float, E0: float, Rsol: float, k: float, alpha: float, name: str = ""):
        if type(simulationID) == str:
            index = self.simulationnames.index(simulationID)
        else:
            index = simulationID
        if simulationID != -1:
            self.simulations[index].addCell(experimentID, x, dx, cinf, D, A, E0, Rsol, k, alpha, name)
        else:
            for simulation in self.simulations:
                simulation.addCell(experimentID, x, dx, cinf, D, A, E0, Rsol, k, alpha, name)

    def modifySimulationParameter(self, simulationID: str | int, param: str, newVal: tuple):
        if type(simulationID) == int:
            index = simulationID
        else:
            index = self.simulationnames.index(simulationID)
        self.simulations[index].modifySimulationParameter(param, newVal)
        if param == "name":
            self.simulationnames[index] = newVal
    
    def modifyExperimentParameter(self, simulationID: str | int, experimentID: str | int, param: str, newVal: float):
        if type(simulationID) == int:
            index = simulationID
        else:
            index = self.simulationnames.index(simulationID)
        self.simulations[index].modifyExperimentParameter(experimentID, param, newVal)

    def modifyCellParameter(self, simulationID: str | int, experimentID: str | int, cellID: str | int, param: str, newVal: float):
        if type(simulationID) == int:
            index = simulationID
        else:
            index = self.simulationnames.index(simulationID)
        self.simulations[index].modifyCellParameter(experimentID, cellID, param, newVal)

    def simulate(self):
        for simulation in self.simulations:
            simulation.simulate()

    def plotSimulations(self):
        for simulation in self.simulations:
            simulation.plotNoShow()
        plt.show()
    
    def plotSimulationsCV(self):
        for simulation in self.simulations:
            simulation.plotCVNoShow()
        plt.show()

    def plotCells(self):
        concProfiles = {}
        currents = {}
        for simulation in self.simulations:
            for experiment in simulation.experiments:
                for cell in experiment.cells:
                    if cell.name not in concProfiles.keys():
                        newConcFigure, newConcSubplot = plt.subplots()
                        newCurrFigure, newCurrSubplot = plt.subplots()
                        newConcFigure.suptitle('Cell ' + cell.name)
                        newCurrFigure.suptitle('Cell ' + cell.name)
                        concProfiles.update({cell.name: [newConcFigure, newConcSubplot]})
                        currents.update({cell.name: [newCurrFigure, newCurrSubplot]})
                    xcoords = np.linspace(0, cell.x, cell.numOfElements, False)
                    concProfiles[cell.name][1].plot(xcoords, cell.cell[0], label = simulation.name + " " + experiment.name + " " + cell.name + " Ox")
                    concProfiles[cell.name][1].plot(xcoords, cell.cell[1], label = simulation.name + " " + experiment.name + " " + cell.name + " Red")
                    tcoords = np.linspace(0, experiment.t, int(experiment.t/experiment.dt), False)
                    currents[cell.name][1].plot(tcoords, cell.current, label = simulation.name + " " + experiment.name + " " + cell.name)
        for concProfile in concProfiles.values():
            concProfile[0].legend()
        for current in currents.values():
            current[0].legend()
        #plt.show()

    def plotAllCV(self):
        CVs = {}
        for simulation in self.simulations:
            for experiment in simulation.experiments:
                for cell in experiment.cells:
                    if cell.name not in CVs.keys():
                        newCVFigure, newCVSubplot = plt.subplots()
                        newCVFigure.suptitle('Cell ' + cell.name)
                        CVs.update({cell.name: [newCVFigure, newCVSubplot]})
                    CVs[cell.name][1].plot(experiment.voltages, cell.current, label = simulation.name + " " + experiment.name + " " + cell.name)
        for CV in CVs.values():
            CV[0].legend()
        #plt.show()

    def showPlots(self):
        plt.show()

    def __str__(self):
        if not self.simulations:
            programString = "No simulations in the program."
            return programString
        programString = "Simulations in this program:\n"
        for simulation in self.simulations:
            programString += "Simulation " + simulation.name + "\n"
        return programString

    def __len__(self):
        return len(self.simulations)

    def __repr__(self):
        return f"{self.__class__.__name__}(number of simulations={len(self.simulations)})"