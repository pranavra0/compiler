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
