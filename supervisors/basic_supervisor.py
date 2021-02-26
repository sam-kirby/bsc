import os
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
u_size = 16 # comm.Get_attr(MPI.UNIVERSE_SIZE)

if rank == 0:
    print(f"Supervisor is running at rank {rank}. The universe contains {u_size} nodes.")
else:
    print(f"Supervisor expected to be running at rank 0, but is actually rank {rank}")
    sys.exit(1)

comms = []

root = os.getcwd()

for i in range(u_size - 1):
    a_0 = (i+1) * 0.125
    dname = f"results{a_0}"
    os.mkdir(dname)

    info = MPI.Info.Create()
    info.Set("wdir", f"{root}/{dname}")

    comms.append(MPI.COMM_SELF.Spawn(
        command = "smilei_sub",
        args=[
            f"a_0 = {a_0}",
            f"{os.environ['HOME']}/namelists/{sys.argv[1]}.py"
        ],
        maxprocs=1,
        info=info
    ))

recvs = [(i, comm.irecv(source=0, tag=0)) for (i, comm) in enumerate(comms)]
while len(recvs) != 0:
    for item in recvs:
        i, recv = item
        if recv.Test():
            recv.Wait()
            print(f"Process {i} completed")
            recvs.remove(item)
    time.sleep(3)

print("No Smilei tasks are running, exiting...")
