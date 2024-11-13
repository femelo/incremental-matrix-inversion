
##### Configurable options:

## Compiler:
CC=gcc
#CC=cc

BLASLIB = -lblas

## Compiler flags:

# GCC:  (also -march=pentium etc, for machine-dependent optimizing)
CFLAGS=-Wall -O3 -fomit-frame-pointer -funroll-loops -fPIC

# GCC w/ debugging:
#CFLAGS=-Wall -g -DINLINE=

# Compaq C / Digital C:
#CFLAGS=-arch=host -fast

# SunOS:
#CFLAGS=-fast

## Program options:

# Enable long options for cl (eg. "cl --help"), comment out to disable.
# Requires getopt_long(3)  (a GNU extension)
LONGOPTS = -DENABLE_LONG_OPTIONS

##### End of configurable options

all: lib

clean:
	rm -f *.o *.so *~

backup:
	mkdir "`date "+backup-%Y-%m-%d-%H-%M"`" 2>/dev/null || true
	cp * "`date "+backup-%Y-%m-%d-%H-%M"`"  2>/dev/null || true

lib: approxInv.so

approxInv.so: approxInv.o
	$(CC) -shared -o $@ $(BLASLIB) approxInv.o

approxInv.o: approxInv.c
	$(CC) $(CFLAGS) -o $@ $(BLASLIB) -c $<
