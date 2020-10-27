# N-body-simulation

- [x] Brute Force method, O(n^2)
- [ ] Object collisions
- [ ] Leapfrog/Verlet integration
- [ ] Visualizer based on Unity Engine
- [ ] Barnes–Hut simulation, O(n log n)
- [ ] Fractional energy error
- [ ] 2D -> 3D
- [ ] GPU kernels + Tensor core ops


### Brute Force | O(n^2)
#### Performance measurements | R9 3900X (12 cores, 24 threads)
|    N    |  100  |  500  | 2000 | 5000 | 10000 | 50000 |
| ------- | :---: | :---: | :--: | :--: | :--: | :--: |
| 1 thread (ticks/sec)| 4018 | 156 | 9.7 | 1.57 | - | - |
| 4 threads (ticks/sec)| - | 450 | 31 | 5.78 | 1.46 | - |
| 8 threads (ticks/sec)| - | - | - | - | - | - |
| 12 threads (ticks/sec)| - | - | - | - | - | - |
| 24 threads (ticks/sec)| - | - | - | - | - | - |
#### Derived functions
P1(N) = --/N^2  ticks/sec  
P4(N) = --/N^2  ticks/sec  
P8(N) = --/N^2  ticks/sec  
P12(N) = --/N^2  ticks/sec  
P24(N) = --/N^2  ticks/sec  

### Barnes–Hut simulation | O(n log n)
