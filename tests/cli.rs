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
