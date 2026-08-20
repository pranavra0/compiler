# Compy

an experimental compiler and systems programming language. Lowers to LLVM IR and can also interpret them without producing a native executable.

The language is aimed at small, explicit programs. fixed-width integers, structs, arrays, slices, raw pointers, modules, compile-time evaluation, C ABI 

## Build the compiler

```sh
cargo build
```

Run the compiler through Cargo while developing:

```sh
cargo run -- run examples/hello.compy
```

The program's return value becomes the process exit code. To build a native executable instead:

```sh
cargo run -- build examples/hello.compy -o hello
./hello
```

`build` uses the native target by default. It writes an intermediate object file next to the executable. Use `-O0`, `-O1`, `-O2`, or `-O3` to select the LLVM optimization level.


Read [HELP.md](HELP.md) for more info 