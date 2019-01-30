CC = g++
LD = g++
LIBS = -lm
LDFLAGS =
SOURCES = $(wildcard *.cc)
OBJECTS = $(SOURCES:.cc=.o)
XEC = hmw01

all: clean a.exe

a.exe: $(OBJECtS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIBS) --o $@

%.o %.cc
	$(CC) $(CFLAGS) $< -o $@

