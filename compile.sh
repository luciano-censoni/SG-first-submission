#!/bin/bash

echo
echo 'starting compilation for linux'
gcc -Wall -fPIC -shared -o lib_time_evo.so c_time_evo.c -lm
echo "compilation for linux successful"
echo
echo
echo 'starting compilation for windows'
x86_64-w64-mingw32-gcc -Wall -fPIC -D FOR_WINDOWS -shared -o lib_time_evo.dll c_time_evo.c -lm
echo "compilation for windows successful"
echo

