
## A small program

```compy
Pair :: struct {
    left: i32;
    right: i32;
}

sum :: (pair: Pair) -> i32 {
    return pair.left + pair.right;
}

main :: () -> i32 {
    pair := Pair{left = 20, right = 22};
    return sum(pair);
}
```

Functions and named declarations use `::`. `:=` infers a local variable's type. Types such as `i32`, `f64`, `bool`, `*T` (pointer), `[]T` (slice), and `[N]T` (array) are written directly in the source.

## Compiler commands

The binary produced by Cargo is named `compiler`.

| Command | Function |
| --- | --- |
| `compiler run <file>` | Interpret a program and return its exit code. `interpret` is an alias. |
| `compiler build <file>` | Compile, emit a native object, and link an executable. |
| `compiler check <file>` | Resolve modules and run semantic analysis without code generation. |
| `compiler ir <file>` | Generate LLVM IR and print it, or write it with `-o`. |
| `compiler lex <file>` | Print the lexer tokens. |
| `compiler parse <file>` | Print the parsed syntax tree. |
| `compiler fmt <files...>` | Format source files. Add `--check` to fail if a file would change. |
| `compiler reflect <file> <type>` | Print compile-time metadata for a type. |
| `compiler reflect <file> function <name>` | Print metadata for a function. |
| `compiler generated <file>` | Print the program after compile-time generation. |

`compile` is an alias for `build`.

## Language features

### Control flow and cleanup

The compiler supports `if` and `else`, `while`, `break`, `continue`, `return`, and `defer`

### Explicit memory operations

Arrays and slices can be indexed, and slices carry a pointer and a length. Normal dynamic indexing is checked at runtime. Use `unchecked_index` when an unchecked access is intentional. Pointers support address-of, dereference, comparison, pointer arithmetic, and pointer distance.

```compy
main :: () -> i32 {
    values := [3]i32{10, 20, 30};
    first := &values[0];
    second := first + 1;
    *second = 21;
    return values[1];
}
```

Layout queries are available through `size_of`, `align_of`, and `offset_of`.

### Result values

A return type can contain a value type and an error type, separated by `|`. `?` propagates an error result to the caller.

```compy
read :: (value: i32) -> i32 | i32 {
    if value < 0 {
        return return_err(1);
    }
    return return_ok(value);
}

main :: () -> i32 {
    value := read(7)?;
    return value;
}
```

Result values can also be inspected with helpers such as `is_err` and `unwrap`.

### Modules and C interop

A module is a `.compy` file imported by filename. An import creates a qualified name, and only declarations marked `export` are visible from another module.

```compy
// math.compy
export add :: (a: i32, b: i32) -> i32 {
    return a + b;
}
```

```compy
// main.compy
import math;

main :: () -> i32 {
    return math.add(2, 3);
}
```

Use `-I` when a module is outside the source file's directory. C declarations and exported symbols can use the C ABI:

```compy
extern "c" abs(value: i32) -> i32;
export "c" add :: (a: i32, b: i32) -> i32 {
    return a + b;
}
```

### Compile-time evaluation

Prefix an expression with `#` to evaluate it during compilation. Compile-time code can calculate constants, inspect type and function metadata, validate layouts, generate declarations, and specialize the supported generic functions and structs.

```compy
Packet :: struct {
    tag: u8;
    payload: i32;
}

packet_size :: #size_of(Packet);
#validate(packet_size > 0, "Packet must have a layout");
#generate_constant("answer", 42);

main :: () -> i32 {
    return answer;
}
```
