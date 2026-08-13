gcc -c main.c json.c safetensors.c config.c && mv *.o build/
gcc build/*.o  -o build/main
