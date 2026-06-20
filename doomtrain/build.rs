use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let doomrl = manifest.parent().unwrap().join("doomrl");
    let build_dir = doomrl.join("build");
    let lib = build_dir.join("libdoomrl.a");

    let status = Command::new("bash")
        .arg(doomrl.join("build.sh"))
        .current_dir(&doomrl)
        .status()
        .expect("failed to run doomrl/build.sh");
    assert!(status.success(), "doomrl/build.sh failed");
    assert!(lib.exists(), "expected {} after build.sh", lib.display());

    println!(
        "cargo:rerun-if-changed={}",
        doomrl.join("doomrl.c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        doomrl.join("doomrl.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        doomrl.join("build.sh").display()
    );

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=doomrl");
}
