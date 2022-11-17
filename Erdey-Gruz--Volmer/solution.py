### Modules
import time
from simulation_classes import Program


start_time = time.time()
### Preamble: basic quantities and constants of the simulation
F = 96485.3     # C/mol
x = 0.1         # cm
dx = 0.001      # cm = 10 um
t = 100         # s
dt = 0.001      # s = 1 ms
D = [1e-6, 1e-6]        # cm2/s
cinf = 1e-6     # mol/cm3
E0 = 0          # V
k = 1e-6        # s-1
alpha = 0.5     # 1
A = 1           # cm2
Rsol = 10       # ohm

program = Program()
program.addSimulation("Built-in exp")
#program.addSimulation("1, 1 Padé", (1, 1))
#program.addSimulation("Explicit Euler", (1, 0))
#program.addSimulation("Implicit Euler", (0, 1))
program.addExperiment(-1, t, dt, -1, 1, 0.05, 0, "Basic Experiment")
#program.addExperiment(-1, 2*t, dt, "Double time")
#program.addExperiment(-1, t, dt*2, "Half time precision")
program.addCell(-1, -1, x, dx, cinf, D, A, E0, Rsol, k, alpha, "Basic cell")
#program.addCell(-1, -1, x, dx, cinf, 10*D, A, E0, Rsol, k, alpha, "Cell with 10x diffusion")
#program.addCell(-1, -1, x, dx/10, cinf, 10*D, "Most accurate cell with 10x diffusion")
#program.addCell(-1, -1, x, dx/2, cinf, D, "More accurate cell")
#program.addCell(-1, -1, x, dx/10, cinf, D, "Most accurate cell")
program.simulate()
print("--- %s seconds ---" % (time.time() - start_time))
#program.plotSimulations()
#program.plotCells()
program.plotAllCV()
program.showPlots()