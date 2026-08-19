use std::path::PathBuf;
use std::process::Command;

#[test]
fn core_language_fixtures_run_through_the_cli() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/language/core");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|x| x.to_str()) != Some("compy") {
            continue;
        }
        let result = Command::new(env!("CARGO_BIN_EXE_compiler"))
            .args(["run", path.to_str().unwrap()])
            .output()
            .unwrap();
        let expected: i32 = std::fs::read_to_string(path.with_extension("expected"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(expected),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn native_backend_agrees_with_interpreter_on_core_fixtures() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/language/core");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|x| x.to_str()) != Some("compy") {
            continue;
        }
        let expected: i32 = std::fs::read_to_string(path.with_extension("expected"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let output = std::env::temp_dir().join(format!(
            "compy_native_{}_{}",
            std::process::id(),
            path.file_stem().unwrap().to_string_lossy()
        ));
        let build = Command::new(env!("CARGO_BIN_EXE_compiler"))
            .args([
                "build",
                path.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&build.stderr)
        );
        assert_eq!(
            Command::new(&output).status().unwrap().code(),
            Some(expected),
            "{}",
            path.display()
        );
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(output.with_extension("o"));
    }
}

#[test]
fn negative_language_fixtures_are_compile_fail_tests() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/language/negative");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|x| x.to_str()) != Some("compy") {
            continue;
        }
        let result = Command::new(env!("CARGO_BIN_EXE_compiler"))
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            !result.status.success(),
            "accepted invalid fixture {}",
            path.display()
        );
        let diagnostic = String::from_utf8_lossy(&result.stderr);
        let expected = std::fs::read_to_string(path.with_extension("expected")).unwrap();
        for stable in expected.lines().filter(|line| !line.trim().is_empty()) {
            assert!(
                diagnostic.contains(stable),
                "{}: missing diagnostic `{stable}` in {diagnostic}",
                path.display()
            );
        }
    }
}
