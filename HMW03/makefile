CC = g++ -fopenmp
LD = g++ -fopenmp
LIBS = -lm
LDFLAGS = 
SOURCES = $(wildcard.cc)
OBJECTS = $(SOURCES:.cc.=.o)
XEC=hmw03
all: clean a.exe
a.exe: $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIBS) --o$@
%.o%.cc
	$(CC) $(CLFAGS) $< -0 $!@
	
