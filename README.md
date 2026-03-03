Run Cython files at the command line by typing:
python3 setup_cython.py build_ext -fi

Run mpi4py files at the command line by typing:

mpirun -n 4 python3 <filename> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

ex: mpirun -n 4 python3 mpi4py_run.py 50 50 0.5 1