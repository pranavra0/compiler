use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn paths(label: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "compiler_cli_{}_{}_{}",
        std::process::id(),
        label,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    (root.with_extension("compy"), root)
}

fn compiler() -> Command {
    Command::new(env!("CARGO_BIN_EXE_compiler"))
}

#[test]
fn compiler_programmability_example_builds_and_exposes_generated_output() {
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("compiler_programmability.compy");
    let (_, output) = paths("programmability");
    let build = compiler()
        .args(["build"])
        .arg(&example)
        .args(["-o"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(0));

    let reflection = compiler()
        .args(["reflect"])
        .arg(&example)
        .arg("Packet")
        .output()
        .unwrap();
    let reflection_stdout = String::from_utf8_lossy(&reflection.stdout);
    assert!(reflection.status.success());
    assert!(reflection_stdout.contains("FieldInfo"));

    let generated = compiler()
        .args(["generated"])
        .arg(&example)
        .output()
        .unwrap();
    let generated_stdout = String::from_utf8_lossy(&generated.stdout);
    assert!(generated.status.success());
    assert!(generated_stdout.contains("generated_packet_size"));

    let _ = fs::remove_file(&output);
    let _ = fs::remove_file(output.with_extension("o"));
}

#[test]
fn check_rejects_invalid_programs() {
    let cases = [
        "main :: () -> i32 { return missing; }",
        "main :: () -> i32 { x :: 1; x = 2; return x; }",
        "main :: () -> i32 { x: i32 = 1; y: i64 = 2; return x + y; }",
    ];

    for (index, source) in cases.iter().enumerate() {
        let (source_path, output_root) = paths(&format!("check_{index}"));
        fs::write(&source_path, source).unwrap();

        let result = compiler().arg("check").arg(&source_path).output().unwrap();
        assert!(
            !result.status.success(),
            "invalid program was accepted: {source}"
        );
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("semantic error"),
            "missing semantic diagnostic for {source}"
        );

        let _ = fs::remove_file(source_path);
        let _ = fs::remove_dir_all(output_root);
    }
}

#[test]
fn diagnostics_include_file_line_and_column() {
    let (source_path, output_root) = paths("diagnostic_location");
    fs::write(
        &source_path,
        "main :: () -> i32 {\n    return missing;\n}\n",
    )
    .unwrap();
    let result = compiler().arg("check").arg(&source_path).output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains(&format!("{}:2:12", source_path.display())),
        "diagnostic had no source location: {stderr}"
    );
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_dir_all(output_root);
}

#[test]
fn build_requires_a_valid_main_entry_point() {
    for (label, source) in [
        ("no_main", "helper :: () {}"),
        ("main_args", "main :: (arg: i32) -> i32 { return arg; }"),
        ("main_return", "main :: () -> i64 { return 0; }"),
        ("main_variable", "main :: 1;"),
    ] {
        let (source_path, output_root) = paths(label);
        fs::write(&source_path, source).unwrap();
        let result = compiler()
            .args(["build"])
            .arg(&source_path)
            .output()
            .unwrap();
        assert!(
            !result.status.success(),
            "invalid entry point accepted: {source}"
        );
        assert!(String::from_utf8_lossy(&result.stderr).contains("semantic error"));
        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(&output_root);
        let _ = fs::remove_file(output_root.with_extension("o"));
    }
}

#[test]
fn build_rejects_semantically_invalid_program() {
    let (source_path, output_root) = paths("build_invalid");
    fs::write(
        &source_path,
        "main :: () -> i32 { return does_not_exist(1); }",
    )
    .unwrap();

    let result = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("semantic error"));

    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn build_executable_preserves_exit_code() {
    let (source_path, output_root) = paths("build_exit");
    fs::write(&source_path, "main :: () -> i32 { return 37; }").unwrap();

    let build = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&output_root).status().unwrap();
    assert_eq!(run.code(), Some(37));

    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn builds_and_runs_core_loop_programs() {
    let cases = [
        (
            "count",
            10,
            "main :: () -> i32 { i := 0; while i < 10 { i = i + 1; } return i; }",
        ),
        (
            "break",
            7,
            "main :: () -> i32 { i := 0; while true { if i == 7 { break; } i = i + 1; } return i; }",
        ),
        (
            "continue",
            50,
            "main :: () -> i32 { i := 0; sum := 0; while i < 10 { i = i + 1; if i == 5 { continue; } sum = sum + i; } return sum; }",
        ),
        (
            "nested",
            6,
            "main :: () -> i32 { total := 0; i := 0; while i < 3 { i = i + 1; j := 0; while j < 3 { j = j + 1; if j == 1 { continue; } break; } total = total + i; } return total; }",
        ),
        (
            "return",
            9,
            "main :: () -> i32 { i := 0; while i < 3 { if i == 1 { return 9; } i = i + 1; } return 0; }",
        ),
        (
            "followed_by_return",
            4,
            "main :: () -> i32 { i := 0; while i < 0 { i = i + 1; } return 4; }",
        ),
    ];

    for (label, expected, source) in cases {
        let (source_path, output_root) = paths(label);
        fs::write(&source_path, source).unwrap();

        let build = compiler()
            .args(["build"])
            .arg(&source_path)
            .arg("-o")
            .arg(&output_root)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{label} build failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );

        let run = Command::new(&output_root).status().unwrap();
        assert_eq!(run.code(), Some(expected), "incorrect result for {label}");

        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(&output_root);
        let _ = fs::remove_file(output_root.with_extension("o"));
    }
}

#[test]
fn builds_structs_arrays_globals_and_constants() {
    let cases = [
        (
            "aggregates",
            52,
            "Pair :: struct { x: i32; y: i32; } limit :: i32 = 40; counter : i32 = 1; main :: () -> i32 { p := Pair{x = 1, y = 2}; xs := [3]i32{4, 5, 6}; p.x = 3; xs[1] = 7; counter = counter + 1; return p.x + xs[1] + limit + counter; }",
        ),
        (
            "rvalue_access",
            9,
            "Pair :: struct { x: i32; } make :: () -> Pair { return Pair{x = 9}; } main :: () -> i32 { return make().x; }",
        ),
        (
            "wrapped_constant",
            42,
            "small :: u8 = 255 + 1; ready : bool = small == 0; main :: () -> i32 { if ready { return 42; } return 0; }",
        ),
    ];

    for (label, expected, source) in cases {
        let (source_path, output_root) = paths(label);
        fs::write(&source_path, source).unwrap();
        let build = compiler()
            .args(["build"])
            .arg(&source_path)
            .arg("-o")
            .arg(&output_root)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{label} build failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&output_root).status().unwrap();
        assert_eq!(run.code(), Some(expected), "incorrect result for {label}");
        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(&output_root);
        let _ = fs::remove_file(output_root.with_extension("o"));
    }
}

#[test]
fn memory_example_exercises_raw_pointers_slices_layout_and_indexing() {
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/memory.compy");
    let output_root =
        std::env::temp_dir().join(format!("compiler_memory_example_{}", std::process::id()));
    let build = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "memory example failed to build: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&output_root).status().unwrap();
    assert_eq!(run.code(), Some(88));
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn checked_memory_index_traps_at_runtime() {
    let source_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/memory_checked_trap.compy");
    let output_root =
        std::env::temp_dir().join(format!("compiler_memory_trap_{}", std::process::id()));
    let build = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();
    assert!(build.status.success(), "trap example failed to build");
    let run = Command::new(&output_root).status().unwrap();
    assert!(!run.success(), "an invalid checked index returned normally");
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn unchecked_index_has_no_generated_bounds_trap() {
    let (source_path, output_root) = paths("unchecked_ir");
    fs::write(
        &source_path,
        "main :: () -> i32 { values := [2]i32{10, 20}; return unchecked_index(values, 1); }",
    )
    .unwrap();
    let ir = compiler().arg("ir").arg(&source_path).output().unwrap();
    assert!(ir.status.success(), "unchecked example failed to lower");
    let text = String::from_utf8_lossy(&ir.stdout);
    assert!(text.contains("getelementptr"));
    assert!(!text.contains("index.valid"));
    assert!(!text.contains("llvm.trap"));
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(output_root);
}

#[test]
fn result_values_propagate_and_are_inspectable() {
    let (source_path, output_root) = paths("result_values");
    fs::write(
        &source_path,
        "read :: (x: i32) -> i32 | i32 { if x == 0 { return return_err(7); } return return_ok(x + 1); } get :: (x: i32) -> i32 | i32 { value := read(x)?; return return_ok(value + 1); } main :: () -> i32 { result := get(0); if is_err(result) { return 7; } return unwrap(result); }",
    )
    .unwrap();
    let build = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "result build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert_eq!(Command::new(&output_root).status().unwrap().code(), Some(7));
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn propagation_requires_a_compatible_error_return_type() {
    let (source_path, output_root) = paths("result_invalid_propagation");
    fs::write(
        &source_path,
        "source :: () -> i32 | i64 { return return_ok(1); } wrapper :: () -> i32 | i32 { value := source()?; return return_ok(value); } main :: () -> i32 { return 0; }",
    )
    .unwrap();
    let result = compiler().arg("check").arg(&source_path).output().unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("propagated error type is incompatible")
    );
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(output_root);
}

#[test]
fn defer_is_lifo_captures_values_and_runs_on_loop_exits_and_propagation() {
    let cases = [
        (
            "defer_lifo_capture",
            21,
            "counter : i32 = 0; one :: (x: i32) -> void { counter = counter * 10 + x; } two :: (x: i32) -> void { counter = counter * 10 + x + 1; } f :: () -> void { x := 1; defer one(x); defer two(x); x = 9; } main :: () -> i32 { f(); return counter; }",
        ),
        (
            "defer_early_return",
            1,
            "counter : i32 = 0; inc :: () -> void { counter = counter + 1; } f :: () -> void { defer inc(); return; } main :: () -> i32 { f(); return counter; }",
        ),
        (
            "defer_loop_exit",
            3,
            "counter : i32 = 0; inc :: () -> void { counter = counter + 1; } main :: () -> i32 { i := 0; while i < 3 { defer inc(); i = i + 1; if i == 1 { continue; } if i == 3 { break; } } return counter; }",
        ),
        (
            "defer_propagation",
            1,
            "counter : i32 = 0; inc :: () -> void { counter = counter + 1; } fail :: () -> i32 | i32 { return return_err(9); } wrapped :: () -> i32 | i32 { defer inc(); value := fail()?; return return_ok(value); } main :: () -> i32 { result := wrapped(); if is_err(result) { return counter; } return 0; }",
        ),
    ];
    for (label, expected, source) in cases {
        let (source_path, output_root) = paths(label);
        fs::write(&source_path, source).unwrap();
        let build = compiler()
            .args(["build"])
            .arg(&source_path)
            .arg("-o")
            .arg(&output_root)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{label} build failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );
        assert_eq!(
            Command::new(&output_root).status().unwrap().code(),
            Some(expected)
        );
        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(&output_root);
        let _ = fs::remove_file(output_root.with_extension("o"));
    }
}

#[test]
fn explicit_allocator_arguments_are_ordinary_library_calls() {
    let (source_path, output_root) = paths("allocator_boundary");
    fs::write(
        &source_path,
        "Allocator :: struct { observed: i32; } alloc :: (allocator: *Allocator, size: usize, alignment: usize) -> []u8 | i32 { if alignment == 0 { return return_err(9); } (*allocator).observed = 1; return return_ok(make_slice(null, size)); } dealloc :: (allocator: *Allocator, memory: []u8, alignment: usize) -> void { (*allocator).observed = 8; } main :: () -> i32 { allocator := Allocator{ observed = 0 }; memory := alloc(&allocator, 0, 8); failed := alloc(&allocator, 0, 0); if !is_err(failed) { return 99; } if is_err(memory) { return 98; } dealloc(&allocator, unwrap(memory), 8); return allocator.observed; }", 
    )
    .unwrap();
    let build = compiler()
        .args(["build"])
        .arg(&source_path)
        .arg("-o")
        .arg(&output_root)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "allocator boundary failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert_eq!(Command::new(&output_root).status().unwrap().code(), Some(8));
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(&output_root);
    let _ = fs::remove_file(output_root.with_extension("o"));
}

#[test]
fn rejects_invalid_loop_programs_before_codegen() {
    let cases = [
        "main :: () -> i32 { break; return 0; }",
        "main :: () -> i32 { continue; return 0; }",
        "main :: () -> i32 { while 1 { break; } return 0; }",
    ];

    for (index, source) in cases.iter().enumerate() {
        let (source_path, output_root) = paths(&format!("loop_invalid_{index}"));
        fs::write(&source_path, source).unwrap();

        let result = compiler().arg("build").arg(&source_path).output().unwrap();
        assert!(!result.status.success(), "invalid loop accepted: {source}");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("semantic error"),
            "missing semantic diagnostic for {source}"
        );

        let _ = fs::remove_file(source_path);
        let _ = fs::remove_file(&output_root);
        let _ = fs::remove_file(output_root.with_extension("o"));
    }
}
