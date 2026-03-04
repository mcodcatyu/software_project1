Please view the complete folder and mpi4py file in mpi4py folder
Run Cython files at the command line by typing:
    python setup_cython.py build_ext -fi
For Cython files:
    1. LLcython.pyx:
        python setup_cython.py build_ext -fi
        python run_cython.py <filename> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
    2. LLcython_pra.pyx:
        python setup_cython.py build_ext -fi
        python run_pra.py <filename> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG><THREAD>

Run mpi4py files at the command line by typing:

    mpirun -n 4 python <filename> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

ex: mpirun -n 4 python mpi4py_run.py 50 50 0.5 1

Run test_file.py at the command line by typing:
pytest test_file.py
python3 -m pytest test_file.py
* To test the different file, please modify the import line