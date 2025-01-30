# NUX and GGML experiments

This is a prototype repository. It's a workspace for random experiments with NUX and GGML.

Currently it supports compiling GGML in kernel mode in NUX.


To run:

   git submodule update --init --recursive
   mkdir build
   cd build
   ../configure ARCH=amd64
   (cd ../contrib/ggmlux/ggml/examples/gpt-2; ./download-ggml-model.sh 117M)
   make qemu

