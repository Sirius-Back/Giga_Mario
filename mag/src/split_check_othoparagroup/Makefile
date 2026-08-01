CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?=

SRC = io.cpp merge_summarize.cpp main.cpp
OBJ = $(SRC:.cpp=.o)
BIN = split_check_othoparagroup

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)
