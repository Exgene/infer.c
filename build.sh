gcc -O3 -march=native -flto -fopenmp -I/usr/include/openblas -c *.c && mv *.o build/
gcc -flto -fopenmp build/*.o -lopenblaso -lm -o build/main
