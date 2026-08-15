gcc -c *.c && mv *.o build/
gcc build/*.o -lm -o build/main
