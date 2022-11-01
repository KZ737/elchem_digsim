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

class Cell:
    def __init__(self, x: float, dx: float, cinf: float, D: float, name: str):
        self.x = x
        self.dx = dx
        self.cinf = cinf
        self.D = D
        self.name = name
        self.createCell()
        #self.numOfElements = int(self.x / self.dx)
        #self.cell = np.full(self.numOfElements, self.cinf)

    def __str__(self):
        return str(self.cell)

    def createCell(self):
        self.numOfElements = int(self.x / self.dx)
        self.cell = np.full(self.numOfElements, self.cinf)

    def electrodeReaction(self):
        self.cell[0] = 0

    def setdt(self, dt: float, padeparams: tuple):
        transformMatrix = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape = (self.numOfElements, self.numOfElements))
        transformMatrix = transformMatrix.tocsc()
        transformMatrix[0, 0] = -1
        transformMatrix[self.numOfElements-1, self.numOfElements-1] = -1
        transformMatrix = self.D * dt / (self.dx**2) * transformMatrix
        # transformMatrix = sp.sparse.linalg.expm(transformMatrix)
        if padeparams[0] == -1:
            transformMatrix = sp.sparse.linalg.expm(transformMatrix)
        else:
            transformMatrix = ea.general_pade(padeparams[0], padeparams[1], transformMatrix)
        self.__transformMatrix = transformMatrix

    def propagate(self):
        self.cell = self.__transformMatrix.dot(self.cell)

    def modifyParameter(self, param: str, newVal: str | float):
        setattr(self, param, newVal)
        self.createCell()

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, dx={self.dx}, cinf={self.cinf}, D={self.D})"


class Experiment:
    def __init__(self, t: float, dt: float, name: str):
        self.t = t
        self.dt = dt
        self.name = name
        self.numOfTimesteps = int(self.t / self.dt)
        self.cells = []
        self.cellnames = []

    def addCell(self, x: float, dx: float, cinf: float, D: float, name: str = "", padeparams: tuple = (-1,)):
        if not name:
            name = str(len(self.cells))
        newCell = Cell(x, dx, cinf, D, name)
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
    
    def electrodeReaction(self):
        for cell in self.cells:
            cell.electrodeReaction()

    def propagate(self):
        for cell in self.cells:
            cell.propagate()

    def simulate(self):
        for i in range(self.numOfTimesteps):
            self.electrodeReaction()
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
            plt.plot(xcoords, cell.cell, label = cell.name)
        plt.legend()
        plt.show()

    def plotNoShow(self, plot: plt.figure):
        for cell in self.cells:
            xcoords = np.linspace(0, cell.x, cell.numOfElements, False)
            plot.plot(xcoords, cell.cell, label = cell.name)
        plot.legend()


class Simulation:
    def __init__(self, name: str, padeparams: tuple = (-1,)):
        self.padeparams = padeparams
        self.name = name
        self.experiments = []
        self.experimentnames = []
    
    def addExperiment(self, t: float, dt: float, name: str = ""):
        if not name:
            name = str(len(self.experiments))
        newExperiment = Experiment(t, dt, name)
        self.experiments.append(newExperiment)
        self.experimentnames.append(name)

    def addCell(self, experimentID: str | int, x: float, dx: float, cinf: float, D: float, name: str = ""):
        if type(experimentID) == str:
            index = self.experimentnames.index(experimentID)
        else:
            index = experimentID
        if experimentID != -1:
            self.experiments[index].addCell(x, dx, cinf, D, name, self.padeparams)
        else:
            for experiment in self.experiments:
                experiment.addCell(x, dx, cinf, D, name, self.padeparams)

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

    def addExperiment(self, simulationID: str | int, t: float, dt: float, name: str = ""):
        if type(simulationID) == str:
            index = self.simulationnames.index(simulationID)
        else:
            index = simulationID
        if simulationID != -1:
            self.simulations[index].addExperiment(t, dt, name)
        else:
            for simulation in self.simulations:
                simulation.addExperiment(t, dt, name)

    def addCell(self, simulationID: str | int, experimentID: str | int, x: float, dx: float, cinf: float, D: float, name: str = ""):
        if type(simulationID) == str:
            index = self.simulationnames.index(simulationID)
        else:
            index = simulationID
        if simulationID != -1:
            self.simulations[index].addCell(experimentID, x, dx, cinf, D, name)
        else:
            for simulation in self.simulations:
                simulation.addCell(experimentID, x, dx, cinf, D, name)

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

    def plot(self):
        for simulation in self.simulations:
            simulation.plotNoShow()
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