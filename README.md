# SensingGames

### Getting started
In a Julia session in the project root:
```
] activate .
using SensingGames
SensingGames.test_<game>_<experiment>()
```
Precompilation may take some time, and your first run will also likely take a bit while Julia JIT-compiles Flux and Zygote. Be patient!
