### Modules
import time
from simulation_classes import Program


start_time = time.time()
### Preamble: basic quantities and constants of the simulation
F = 96485.3     # C/mol
x = 0.5         # cm
dx = 0.001      # cm = 10 um
t = 400         # s
dt = 0.01      # s = 1 ms
D = [1e-5, 1e-5]        # cm2/s
cinf = 1e-6     # mol/cm3
E0 = 0          # V
k = 1e-5        # s-1
alpha = 0.5     # 1
A = 1           # cm2
Rsol = 27       # ohm

program = Program()
program.addSimulation("Built-in exp")
#program.addSimulation("1, 1 Padé", (1, 1))
#program.addSimulation("Explicit Euler", (1, 0))
#program.addSimulation("Implicit Euler", (0, 1))
program.addExperiment(-1, t, dt, -0.5, 0.5, 0.01, 0, "Basic Experiment")
#program.addExperiment(-1, 2*t, dt, "Double time")
#program.addExperiment(-1, t, dt*2, "Half time precision")
program.addCell(-1, -1, x, dx, cinf, D, A, E0, 0, k, alpha, "Basic cell 0 res")
#program.addCell(-1, -1, x, dx, cinf, D, A, E0, Rsol, k, alpha, "Basic cell 10 ohm res")
#program.addCell(-1, -1, x, dx, cinf, 10*D, A, E0, Rsol, k, alpha, "Cell with 10x diffusion")
#program.addCell(-1, -1, x, dx/10, cinf, 10*D, "Most accurate cell with 10x diffusion")
#program.addCell(-1, -1, x, dx/2, cinf, D, "More accurate cell")
#program.addCell(-1, -1, x, dx/10, cinf, D, "Most accurate cell")
program.simulate()
print("--- %s seconds ---" % (time.time() - start_time))
#program.plotSimulations()
#program.plotCells()
program.plotAllCVInExp()
program.showPlots()