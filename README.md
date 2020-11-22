# N-body-simulation

- [x] Brute Force method, O(n^2)
- [x] Object collisions
- [ ] Galaxy and Star system initializer
- [ ] Star Density maps + presets + creation tools
- [x] Leapfrog/Verlet integration
- [ ] Visualizer based on Unity Engine
- [ ] Barnes–Hut simulation, O(n log n)
- [ ] Fractional energy error
- [ ] 2D -> 3D
- [ ] GPU kernels + Tensor core ops


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
