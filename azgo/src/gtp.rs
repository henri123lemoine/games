//! Minimal GTP client for strength-calibrated opponents (GNU Go at a given
//! `--level`). One engine process per game; the engine is stateful, so the
//! [`crate::eval::GnuGoAgent`] wrapper keeps its board in sync move by move.
//! Reads go through a forwarding thread so a hung engine times out instead
//! of wedging an unattended run forever.

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

const READ_TIMEOUT: Duration = Duration::from_secs(60);

pub struct Gtp {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Gtp {
    /// Spawns GNU Go at `level` on a 9×9 board with the lab's go rules:
    /// area scoring (`--chinese-rules`), komi 7.5, and `--capture-all-dead`
    /// so its passes agree with our no-dead-stone-adjudication scoring.
    /// `seed` varies its move choice between otherwise identical games.
    pub fn spawn_gnugo(path: &str, level: u32, seed: u32) -> io::Result<Gtp> {
        let mut child = Command::new(path)
            .args([
                "--mode",
                "gtp",
                "--level",
                &level.to_string(),
                "--chinese-rules",
                "--capture-all-dead",
                "--never-resign",
                "--seed",
                &seed.to_string(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("gtp stdin");
        let stdout = child.stdout.take().expect("gtp stdout");
        let (tx, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut gtp = Gtp {
            child,
            stdin,
            lines,
        };
        gtp.cmd("boardsize 9")?;
        gtp.cmd("clear_board")?;
        gtp.cmd("komi 7.5")?;
        Ok(gtp)
    }

    /// Sends one GTP command and returns the success payload (the text after
    /// `= `). A `? ...` failure response becomes an error.
    pub fn cmd(&mut self, command: &str) -> io::Result<String> {
        writeln!(self.stdin, "{command}")?;
        self.stdin.flush()?;
        let mut response: Option<String> = None;
        loop {
            match self.lines.recv_timeout(READ_TIMEOUT) {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if response.is_none() {
                        if trimmed.is_empty() {
                            continue;
                        }
                        response = Some(trimmed);
                    } else if trimmed.is_empty() {
                        // GTP responses end with a blank line.
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!("engine silent for {READ_TIMEOUT:?} on '{command}'"),
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "engine exited",
                    ));
                }
            }
        }
        let response = response.expect("loop exits only after a response line");
        match response.strip_prefix('=') {
            Some(rest) => Ok(rest.trim().to_string()),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gtp failure for '{command}': {response}"),
            )),
        }
    }
}

impl Drop for Gtp {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}
