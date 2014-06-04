LIBS=`pkg-config --libs opencv`
INCLUDE=`pkg-config --cflags-only-I opencv`

all:
	g++ main.cc dr.cc $(INCLUDE) $(LIBS);
debug:
	g++ -g -ggdb main.cc dr.cc $(INCLUDE) $(LIBS);
