gcc -O3 -march=native -flto -fopenmp -c *.c && mv *.o build/
gcc -flto -fopenmp build/*.o -lm -o build/main
