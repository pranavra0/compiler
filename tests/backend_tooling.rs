use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn compiler() -> Command {
    Command::new(env!("CARGO_BIN_EXE_compiler"))
}
fn temp(label: &str) -> (PathBuf, PathBuf) {
    let base = std::env::temp_dir().join(format!("compy_backend_{label}_{}", std::process::id()));
    (base.with_extension("compy"), base)
}

#[test]
fn artifact_and_optimization_controls_are_explicit() {
    let (source, base) = temp("artifacts");
    fs::write(&source, "main :: () -> i32 { return 3; }\n").unwrap();
    let ir = base.with_extension("ll");
    let result = compiler()
        .args([
            "build",
            source.to_str().unwrap(),
            "-O3",
            "--emit-ir",
            "-o",
            ir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(ir.is_file());
    let object = base.with_extension("o");
    let result = compiler()
        .args([
            "build",
            source.to_str().unwrap(),
            "-O0",
            "--emit-obj",
            "-o",
            object.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(object.is_file());
    assert!(String::from_utf8_lossy(&result.stdout).contains("object:"));
    for path in [&source, &ir, &object] {
        let _ = fs::remove_file(path);
    }
}

#[test]
fn debug_ir_contains_source_functions_locals_and_locations() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/debug.compy");
    let (_, base) = temp("debug");
    let ir = base.with_extension("ll");
    let result = compiler()
        .args([
            "build",
            source.to_str().unwrap(),
            "-g",
            "--emit-ir",
            "-o",
            ir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(&ir).unwrap();
    assert!(text.contains("DICompileUnit") && text.contains("DISubprogram"));
    assert!(text.contains("DILocalVariable") && text.contains("DILocation(line: 3"));
    let _ = fs::remove_file(ir);
}

#[test]
fn interpreter_honors_selected_target_layout() {
    let (source, _) = temp("interpreter_target");
    fs::write(&source, "main :: () -> usize { return size_of(*u8); }\n").unwrap();
    let result = compiler()
        .args([
            "run",
            source.to_str().unwrap(),
            "--target",
            "wasm32-unknown-unknown",
        ])
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let _ = fs::remove_file(source);
}

#[test]
fn formatter_formats_and_check_detects_drift() {
    let (source, _) = temp("format");
    fs::write(
        &source,
        "// keep\nmain::()->i32{if true{return 1;}return 0;}\n",
    )
    .unwrap();
    let result = compiler()
        .args(["fmt", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let formatted = fs::read_to_string(&source).unwrap();
    assert!(formatted.starts_with("// keep\n") && formatted.contains("    return 1;"));
    let result = compiler()
        .args(["fmt", "--check", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::write(&source, formatted.replace("    return", "return")).unwrap();
    let result = compiler()
        .args(["fmt", "--check", source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let _ = fs::remove_file(source);
}

#[test]
fn public_examples_compile_as_objects() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut examples: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("compy"))
        .collect();
    examples.push(root.join("modules/main.compy"));
    for (index, source) in examples.iter().enumerate() {
        let output =
            std::env::temp_dir().join(format!("compy_example_{}_{}.o", std::process::id(), index));
        let result = compiler()
            .args([
                "build",
                source.to_str().unwrap(),
                "--emit-obj",
                "-o",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}: {}",
            source.display(),
            String::from_utf8_lossy(&result.stderr)
        );
        let _ = fs::remove_file(output);
    }
}

#[test]
fn invalid_targets_are_reported_without_building() {
    let (source, base) = temp("invalid_target");
    fs::write(&source, "main :: () -> i32 { return 0; }\n").unwrap();
    let result = compiler()
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "not-a-real-target",
            "--emit-obj",
            "-o",
            base.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("target") || error.contains("triple"));
    let _ = fs::remove_file(source);
    let _ = fs::remove_file(base);
}
