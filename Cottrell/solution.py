### Modules
import time
from simulation_classes import Program


start_time = time.time()
### Preamble: basic quantities and constants of the simulation
F = 96485.3     # C/mol
x = 0.1         # cm
dx = 0.01      # cm = 10 um
t = 10          # s
dt = 0.001      # s = 1 ms
D = 1e-6        # cm2/s
cinf = 1e-6     # mol/cm3

program = Program()
program.addSimulation("Simulation built-in exp")
program.addSimulation("Simulation 1, 1 Padé")
program.addExperiment(-1, t, dt, "Basic Experiment")
program.addExperiment(-1, 2*t, dt, "Double time")
program.addExperiment(-1, t, dt*2, "Half time precision")
program.addCell(-1, -1, x, dx, cinf, D, "Basic cell")
program.addCell(-1, -1, x, dx, cinf, 10*D, "Cell with 10x diffusion")
program.addCell(-1, -1, x, dx/10, cinf, 10*D, "Most accurate cell with 10x diffusion")
program.addCell(-1, -1, x, dx/2, cinf, D, "More accurate cell")
program.addCell(-1, -1, x, dx/10, cinf, D, "Most accurate cell")
'''
newSimulation = Simulation("Simulation")
newSimulation.addExperiment(t, dt, "Basic experiment")
newSimulation.addExperiment(2*t, dt, "Double time")
newSimulation.addExperiment(t, dt*2, "Half time precision")
newSimulation.addCell(-1, x, dx, cinf, D, "Basic cell")
newSimulation.addCell(-1, x, dx, cinf, 10*D, "Cell with 10x diffusion")
newSimulation.addCell(-1, x, dx/10, cinf, 10*D, "Most accurate cell with 10x diffusion")
newSimulation.addCell(-1, x, dx/2, cinf, D, "More accurate cell")
newSimulation.addCell(-1, x, dx/10, cinf, D, "Most accurate cell")
newSimulation.addCell("Basic experiment", 2*x, dx, cinf, D, "Double length cell")

newSimulation1 = Simulation("Simulation", (1, 1))
newSimulation1.addExperiment(t, dt, "Basic experiment")
newSimulation1.addExperiment(2*t, dt, "Double time")
newSimulation1.addExperiment(t, dt*2, "Half time precision")
newSimulation1.addCell(-1, x, dx, cinf, D, "Basic cell")
newSimulation1.addCell(-1, x, dx, cinf, 10*D, "Cell with 10x diffusion")
newSimulation1.addCell(-1, x, dx/10, cinf, 10*D, "Most accurate cell with 10x diffusion")
newSimulation1.addCell(-1, x, dx/2, cinf, D, "More accurate cell")
newSimulation1.addCell(-1, x, dx/10, cinf, D, "Most accurate cell")
newSimulation1.addCell("Basic experiment", 2*x, dx, cinf, D, "Double length cell")
'''
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
'''
newSimulation.simulate()
newSimulation1.simulate()
'''
program.simulate()
print("--- %s seconds ---" % (time.time() - start_time))
program.plot()
'''
newSimulation.plot()
newSimulation1.plot()
'''