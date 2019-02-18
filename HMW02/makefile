CC = mpiCC
LD = g++
LIBS = -lm
LDFLAGS =
SOURCES = $(wildcard *.cc)
OBJECTS = $(SOURCES:.cc=.o)
XEC = hmw02

all: clean a.exe

a.exe: $(OBJECtS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIBS) --o $@

%.o %.cc
	$(CC) $(CFLAGS) $< -o $@

