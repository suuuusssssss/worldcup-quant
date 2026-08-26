CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -pthread
TARGET    = mc_tournament

all: $(TARGET)

$(TARGET): mc_tournament.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

debug: mc_tournament.cpp
	$(CXX) -O0 -g -std=c++17 -fsanitize=address,undefined -pthread -o $(TARGET)_debug $<

clean:
	rm -f $(TARGET) $(TARGET)_debug

.PHONY: all debug clean
