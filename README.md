# SensingGames

### Getting started
In a Julia session in the project root:
```
] activate .
using SensingGames
SensingGames.test_localization() # or some other example
```
Precompilation may take some time, and your first run will also likely take a bit while Julia JIT-compiles Flux and Zygote. Be patient!

### Known problems
* We're currently using Plots.jl over Makie.jl. Annoyingly Plots will intercept your interrupts, so it can be difficult to stop easy games where the gradient calculations take little time compared to the plotting.
* When running in parallel gradient calculation occasionally fails. This may be a concurrency bug with States, but the actual error reported is inconsistent. For now we just warn that this has happened and continue.
