import LLcython_pra
import numpy as np
import sys
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREAD = int(sys.argv[5])
        LLcython_pra.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREAD)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG><THREAD>".format(sys.argv[0]))