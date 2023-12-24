mkdir build
cd build || exit 228
cmake ..
cmake --build .
cp LAB3 ../executable
cd ..
rm -r ./build
