# N-body-simulation

- [x] Brute Force method, O(n^2)
- [ ] Barnes–Hut simulation, O(n log n)

### Brute Force
P(N) = 7187096/N^2  ticks/sec  
|    N    |  100  |  500  | 2000 | 10000 | 50000 |
| ------- | :---: | :---: | :--: | :--: | :--: |
| 1 core (ticks/sec)| 719 | 28.7 | 1.82 | - | - |
| 6 cores (ticks/sec)| - | - | - | - | - |
| 12 cores (ticks/sec)| - | - | - | - | - |
