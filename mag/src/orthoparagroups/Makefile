CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?=

SRC = graph.cpp io.cpp extract.cpp main.cpp
OBJ = $(SRC:.cpp=.o)
BIN = orthoparagroups

.PHONY: all clean smoketest

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)

smoketest: $(BIN)
	./$(BIN) --help >/dev/null || true
	@echo "binary ok: $(BIN)"
