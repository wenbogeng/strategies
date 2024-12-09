# Define compiler
CXX=g++

# Define compiler flags
CXXFLAGS=-Wall -std=c++11 -g

# Define the output executable name
TARGET=main

# Default target
all: $(TARGET)

# Rule for building the executable
$(TARGET): main.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.o

# Rule for compiling the source file
main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

# Clean target for removing build artifacts
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all clean
