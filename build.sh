# Configure command with YASM as example and LD as linker
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# Build command
cmake --build build -j
