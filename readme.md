# Overview
This repo is a subset of my home directory on HPC.

## Directories
- jobs: PBS job files
- namelists: Smilei namelists
- patches: My changes to Smilei generated using `git format-patches`
- supervisors: MPI enabled supervisors for running multiple different Smilei processes with different namelists

## Smilei Compilation and Changes
The jobs `compile_smilei` and `compile_smilei_sub` both compile Smilei against Python from the user's base Conda environment and with optimisations enabled for the native CPU. This includes the AVX2 extension.

These jobs currently use the GCC compiler as this seemed to produce a faster binary for running on AMD EPYC CPUs, as used on the general queue. This may need to be revisited if we switch to running on the long high performance queue as this queue is formed of servers running Intel Xeon CPUs.

`compile_smilei` produces unmodified `smilei` and `smilei_test` executables. `compile_smilei_sub` produces the `smilei_sub` executable. `smilei_sub` can be used just like `smilei`, however it can also be managed by a supervisor. This is because, when it closes, it sends a single MPI message to its parent, signalling exit.

These jobs can both be used in the case of updates to Smilei upstream if there are no merge conflicts.

Both these jobs copy the resulting binaries to `~/.local/bin`, which is assumed to exist. Other jobs assume this to be on your `PATH`.

## Supervisors
There are currently two supervisors available:--
- basic
- ml

### basic
The basic supervisor was intended as a proof of concept. As written, it is meant to run on 2 nodes of the general queue (32c/node) with 8 MPI processes per node and 4 threads per process. This is reflected in the `basic_supervisor_test` job.

```sh
#PBS -lselect=2:ncpus=32:mem=16gb:mpiprocs=8:ompthreads=4
```
This gives an MPI universe size of 16 ranks.

Python runs on rank 0 and coordinates the 15 other ranks, which each run a single Smilei simulation where `a0` is varied (`a0 = rank * 0.25`).

This could easily have been achieved through other means - array jobs or sequential Smilei runs - but provides a proof of concept for a method to run Smilei using parameters determined at runtime and affected by the results of other Smilei sims. It is also the only way to run multiple Smilei simulations in parallel in the same job context (I believe, anyhow), as this allows us to start multiple Smilei processes that each believe they are at rank 0.

### ml
The ml supervisor is an example of where the method developed in the basic supervisor is useful. Here we need to run thousands of Smilei simulations where the parameters used will be determined at runtime and will be dictated by the results of past simulations. In addition, it would be ideal if we could run multiple simulations in parallel.

This supervisor is commented and provides detailed logging.

#### Running
First compile `smilei_sub`. Modify GE Parameters section as desired. Create a suitable job script; `ml_supervisor_density.pbs` is a good starting point, and shouldn't need much modification. Submit the job.

Make sure you correct the paths, as your environment will probably not be the same as mine.
