### Modules
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

class Cell:
    def __init__(self, x: float, dx: float, cinf: float, D: float, name: str):
        self.x = x
        self.dx = dx
        self.cinf = cinf
        self.D = D
        self.name = name
        self.numOfCells = int(self.x / self.dx)
        self.cell = np.full(self.numOfCells, self.cinf)

    def __str__(self):
        return str(self.cell)

    def createCell(self):
        self.numOfCells = int(self.x / self.dx)
        self.cell = np.full(self.numOfCells, self.cinf)

    def electrodeReaction(self):
        self.cell[0] = 0

    def setdt(self, dt: float):
        transformMatrix = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape = (self.numOfCells, self.numOfCells))
        transformMatrix = transformMatrix.tocsc()
        transformMatrix[0, 0] = -1
        transformMatrix[self.numOfCells-1, self.numOfCells-1] = -1
        transformMatrix = self.D * dt / (self.dx**2) * transformMatrix
        transformMatrix = sp.sparse.linalg.expm(transformMatrix)
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

    def addCell(self, x: float, dx: float, cinf: float, D: float, name: str = ""):
        if not name:
            name = str(len(self.cells))
        newCell = Cell(x, dx, cinf, D, name)
        newCell.setdt(self.dt)
        self.cells.append(newCell)
        self.cellnames.append(name)

    def modifyCellParameter(self, cellID: str | int, param: str, newVal: str | float):
        if type(cellID) == int:
            index = cellID
        else:
            index = self.cellnames.index(cellID)
        self.cells[index].modifyParameter(param, newVal)
        if param == "name":
            self.cellnames[index] = newVal
        else:
            self.cells[index].setdt(self.dt)

    def modifyExperimentParameter(self, param: str, newVal: str | float):
        setattr(self, param, newVal)
        if param == "t":
            self.numOfTimesteps = int(self.t / self.dt)
        elif param == "dt":
            self.numOfTimesteps = int(self.t / self.dt)
            for cell in self.cells:
                cell.setdt(self.dt)
    
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
            xcoords = np.linspace(0, cell.x, cell.numOfCells, False)
            plt.plot(xcoords, cell.cell, label = cell.name)
        plt.legend()
        plt.show()

    def plotNoShow(self, plot: plt.figure):
        for cell in self.cells:
            xcoords = np.linspace(0, cell.x, cell.numOfCells, False)
            plot.plot(xcoords, cell.cell, label = cell.name)
        plot.legend()


class Simulation:
    def __init__(self, name: str):
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
            self.experiments[index].addCell(x, dx, cinf, D, name)
        else:
            for experiment in self.experiments:
                experiment.addCell(x, dx, cinf, D, name)

    def modifyExperimentParameter(self, experimentID: str | int, param: str, newVal: float):
        if type(experimentID) == int:
            index = experimentID
        else:
            index = self.experimentnames.index(experimentID)
        self.experiments[index].modifyExperimentParameter(param, newVal)
        if param == "name":
            self.experimentnames[index] = newVal

    def modifyCellParameter(self, experimentID: str | int, cellID: str | int, param: str, newVal: float):
        if type(experimentID) == int:
            index = experimentID
        else:
            index = self.experimentnames.index(experimentID)
        self.experiments[index].modifyCellParameter(cellID, param, newVal)

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
            figure.suptitle(repr(experiment))
            subplot = figure.add_subplot()
            experiment.plotNoShow(subplot)
        plt.show()
        


start_time = time.time()
### Preamble: basic quantities and constants of the simulation
F = 96485.3     # C/mol
x = 0.1         # cm
dx = 0.01      # cm = 10 um
t = 10          # s
dt = 0.001      # s = 1 ms
D = 1e-6        # cm2/s
cinf = 1e-6     # mol/cm3


newSimulation = Simulation("Simulation")
newSimulation.addExperiment(t, dt, "Basic experiment")
newSimulation.addExperiment(2*t, dt, "Double time")
newSimulation.addExperiment(t, dt*2, "Half time precision")
newSimulation.addCell(-1, x, dx, cinf, D, "Basic cell")
newSimulation.addCell(-1, x, dx, cinf, 10*D, "Cell with 10x diffusion")
newSimulation.addCell(-1, x, dx/2, cinf, D, "More accurate cell")
newSimulation.addCell(-1, x, dx/10, cinf, D, "Most accurate cell")
newSimulation.addCell("Basic experiment", 2*x, dx, cinf, D, "Double length cell")
#print(newSimulation)
#print(newSimulation.experiments)
'''
newExperiment = Experiment(t, dt, "asd")
newExperiment.addCell(x, dx, cinf, D, "Basic cell")
newExperiment.addCell(x, dx, cinf, 10*D, "Cell with 10x diffusion")
#newExperiment.addCell(x, dx, cinf/2, D/2, "\"Half\" cell")
#newExperiment.modifyCellParameter("Cell with 10x diffusion", "D", D)
#newExperiment.modifyCellParameter("Basic cell", "x", 1)
newExperiment.addCell(x, dx/2, cinf, D, "More accurate cell")
newExperiment.addCell(x, dx/10, cinf, D, "Most accurate cell")
newExperiment.simulate()
'''
newSimulation.simulate()
print("--- %s seconds ---" % (time.time() - start_time))
newSimulation.plot()