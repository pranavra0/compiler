use compiler::modules;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "compiler_modules_{}_{}_{}",
        std::process::id(),
        label,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}
fn compiler() -> Command {
    Command::new(env!("CARGO_BIN_EXE_compiler"))
}

#[test]
fn qualified_imports_and_exports_build() {
    let dir = temp_dir("qualified");
    fs::write(
        dir.join("math.compy"),
        "export \"c\" add :: (a: i32, b: i32) -> i32 { return a + b; }",
    )
    .unwrap();
    fs::write(
        dir.join("main.compy"),
        "import math; main :: () -> i32 { return math.add(2, 3); }",
    )
    .unwrap();
    let output = dir.join("program");
    let result = compiler()
        .args([
            "build",
            dir.join("main.compy").to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(5));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn module_graph_exposes_stable_module_and_declaration_identities() {
    let dir = temp_dir("graph");
    fs::write(
        dir.join("lib.compy"),
        "export first :: () -> i32 { return 1; } export second :: () -> i32 { return 2; }",
    )
    .unwrap();
    fs::write(
        dir.join("main.compy"),
        "import lib; main :: () -> i32 { return lib.first() + lib.second(); }",
    )
    .unwrap();

    let project = modules::resolve(dir.join("main.compy"), &[]).unwrap();
    assert_eq!(project.graph.modules.len(), 2);
    assert_eq!(project.graph.root, project.graph.modules[1].id);
    assert_eq!(project.graph.modules[0].imports.len(), 0);
    assert_eq!(
        project.graph.modules[1].imports[0].target,
        project.graph.modules[0].id
    );
    assert_eq!(project.graph.definitions.len(), 3);
    assert_eq!(
        project
            .graph
            .definitions
            .iter()
            .map(|definition| definition.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        project.graph.definitions.len()
    );
    assert!(project.graph.definitions[0].module != project.graph.root);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn private_imports_are_rejected_and_dependencies_are_reported() {
    let dir = temp_dir("private");
    fs::write(dir.join("lib.compy"), "hidden :: () -> i32 { return 1; }").unwrap();
    fs::write(
        dir.join("main.compy"),
        "import lib; main :: () -> i32 { return lib.hidden(); }",
    )
    .unwrap();
    let result = compiler()
        .args(["check", dir.join("main.compy").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("not exported"));
    fs::write(
        dir.join("main.compy"),
        "import lib; main :: () -> i32 { return 0; }",
    )
    .unwrap();
    let depfile = dir.join("program.d");
    let output = dir.join("program");
    let result = compiler()
        .args([
            "build",
            dir.join("main.compy").to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--depfile",
            depfile.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let deps = fs::read_to_string(depfile).unwrap();
    assert!(deps.contains("lib.compy") && deps.contains("main.compy"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn target_selection_changes_pointer_layout_in_ir() {
    let dir = temp_dir("target");
    let source = dir.join("target.compy");
    fs::write(&source, "main :: () -> usize { return size_of(*u8); }").unwrap();
    let result = compiler()
        .args([
            "ir",
            source.to_str().unwrap(),
            "--target",
            "wasm32-unknown-unknown",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let ir = String::from_utf8_lossy(&result.stdout);
    assert!(ir.contains("p:32:32"), "missing 32-bit target layout: {ir}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_c_objects_and_libraries_are_linked() {
    let dir = temp_dir("link_options");
    fs::write(dir.join("helper.c"), "int helper(void) { return 6; }\n").unwrap();
    let helper_o = dir.join("helper.o");
    let result = Command::new("clang")
        .args([
            "-c",
            dir.join("helper.c").to_str().unwrap(),
            "-o",
            helper_o.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    fs::write(
        dir.join("main.compy"),
        "extern \"c\" helper() -> i32; main :: () -> i32 { return helper(); }",
    )
    .unwrap();
    let output = dir.join("program");
    let result = compiler()
        .args([
            "build",
            dir.join("main.compy").to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--object",
            helper_o.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(6));
    let archive = dir.join("libhelper.a");
    let result = Command::new("ar")
        .args([
            "crus",
            archive.to_str().unwrap(),
            helper_o.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    let library_output = dir.join("library-program");
    let result = compiler()
        .args([
            "build",
            dir.join("main.compy").to_str().unwrap(),
            "-o",
            library_output.to_str().unwrap(),
            "-L",
            dir.to_str().unwrap(),
            "-l",
            "helper",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        Command::new(&library_output).status().unwrap().code(),
        Some(6)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn c_struct_layout_and_abi_match_a_c_fixture() {
    let dir = temp_dir("c_layout");
    fs::write(dir.join("lib.compy"), "export Pair :: struct { x: i32; y: i32; } export \"c\" make :: () -> Pair { return Pair{x = 7, y = 8}; }").unwrap();
    let object = dir.join("lib.o");
    let result = compiler()
        .args([
            "build",
            dir.join("lib.compy").to_str().unwrap(),
            "--emit-object",
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
    fs::write(dir.join("main.c"), "#include <stdint.h>\ntypedef struct { int32_t x; int32_t y; } Pair; extern Pair make(void); int main(void) { Pair p = make(); return p.x != 7 || p.y != 8 || sizeof(Pair) != 8; }\n").unwrap();
    let result = Command::new("clang")
        .args([
            dir.join("main.c").to_str().unwrap(),
            object.to_str().unwrap(),
            "-o",
            dir.join("c-harness").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        Command::new(dir.join("c-harness"))
            .status()
            .unwrap()
            .success()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_cycles_and_duplicate_imports_fail_cleanly() {
    let dir = temp_dir("bad_imports");
    fs::write(
        dir.join("main.compy"),
        "import missing; main :: () -> i32 { return 0; }",
    )
    .unwrap();
    let result = compiler()
        .args(["check", dir.join("main.compy").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("was not found"));
    fs::write(
        dir.join("lib.compy"),
        "import main; export \"c\" x :: () -> i32 { return 0; }",
    )
    .unwrap();
    fs::write(
        dir.join("main.compy"),
        "import lib; main :: () -> i32 { return 0; }",
    )
    .unwrap();
    let result = compiler()
        .args(["check", dir.join("main.compy").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("cycle"));
    fs::write(
        dir.join("lib.compy"),
        "export \"c\" x :: () -> i32 { return 0; }",
    )
    .unwrap();
    fs::write(
        dir.join("main.compy"),
        "import lib; import lib; main :: () -> i32 { return 0; }",
    )
    .unwrap();
    let result = compiler()
        .args(["check", dir.join("main.compy").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("duplicate module alias"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn exported_object_can_be_called_from_c() {
    let dir = temp_dir("ffi");
    fs::write(
        dir.join("lib.compy"),
        "export \"c\" add :: (a: i32, b: i32) -> i32 { return a + b; }",
    )
    .unwrap();
    let object = dir.join("lib.o");
    let result = compiler()
        .args([
            "build",
            dir.join("lib.compy").to_str().unwrap(),
            "--emit-object",
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
    fs::write(dir.join("main.c"), "#include <stdint.h>\nextern int32_t add(int32_t, int32_t); int main(void) { return add(4, 5) != 9; }\n").unwrap();
    let result = Command::new("clang")
        .args([
            dir.join("main.c").to_str().unwrap(),
            object.to_str().unwrap(),
            "-o",
            dir.join("c-harness").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        Command::new(dir.join("c-harness"))
            .status()
            .unwrap()
            .success()
    );
    fs::remove_dir_all(dir).unwrap();
}
