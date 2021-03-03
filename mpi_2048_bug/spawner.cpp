#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm controller;
    MPI_Comm_get_parent(&controller);

    if (controller == MPI_COMM_NULL)
    {
        std::cout << "This is the controller" << std::endl;
        std::ofstream file;
        file.open("out.txt");
        for (int i = 0; i < 4096; i++)
        {
            file << i << std::endl;

            MPI_Comm inter;
            MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &inter, MPI_ERRCODES_IGNORE);

            MPI_Barrier(inter);
            MPI_Comm_disconnect(&inter);
        }
        file.close();
    }
    else
    {
        MPI_Barrier(controller);
        MPI_Comm_disconnect(&controller);
    }

    MPI_Finalize();

    return 0;
}
