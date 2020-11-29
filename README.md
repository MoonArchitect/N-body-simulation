# N-body-simulation

### Integration methods
- Euler
- Euler Symplectic
- Verlet/Leapfrog
- Forest-Ruth
- PEFRL (Position Extended Forest-Ruth Like)

## Brute Force | O(n^2)
#### Performance measurements | R9 3900X (12 cores, 24 threads, 4.1 GHz)
|    N    |  100  |  500  | 2000 | 5000 | 10000 | 50000 |
| ------- | :---: | :---: | :--: | :--: | :--: | :--: |
| ⠀1 thread (ticks/sec)| - | - | - | - | - | - |
| ⠀4 threads (ticks/sec)| - | - | - | - | - | - |
| ⠀8 threads (ticks/sec)| - | - | - | - | - | - |
| 12 threads (ticks/sec)| - | - | - | - | - | - |
| 24 threads (ticks/sec)| - | - | - | - | - | - |

## Barnes–Hut simulation | O(n log n)
\-

### TODO
- [ ] Multithreading for Brute Force method
- [ ] Barnes–Hut simulation, O(n log n)
- [ ] Object collisions
- [ ] Galaxy and Star system initializer
- [ ] Star Density maps + presets + creation tools
- [ ] 3D simulation
- [ ] GPU kernels + Tensor core ops
