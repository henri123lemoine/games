//! Minimal UCI client for strength-calibrated opponents (Stockfish with
//! `UCI_LimitStrength`/`UCI_Elo`). One engine process per game; positions
//! are sent as FEN, so the client stays stateless per move.

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct Uci {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl Uci {
    /// Spawns `path` and configures it to play at `elo` (clamped by the
    /// engine to its supported range; Stockfish 17: 1320–3190).
    pub fn spawn(path: &str, elo: u32) -> io::Result<Uci> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("uci stdin");
        let stdout = child.stdout.take().expect("uci stdout");
        let mut uci = Uci {
            child,
            stdin,
            reader: BufReader::new(stdout),
        };
        uci.send("uci")?;
        uci.wait_for("uciok")?;
        uci.send("setoption name UCI_LimitStrength value true")?;
        uci.send(&format!("setoption name UCI_Elo value {elo}"))?;
        uci.send("isready")?;
        uci.wait_for("readyok")?;
        Ok(uci)
    }

    pub fn best_move(&mut self, fen: &str, movetime_ms: u32) -> io::Result<String> {
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go movetime {movetime_ms}"))?;
        let line = self.wait_for("bestmove")?;
        line.split_whitespace()
            .nth(1)
            .map(str::to_string)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad: {line}")))
    }

    fn send(&mut self, cmd: &str) -> io::Result<()> {
        writeln!(self.stdin, "{cmd}")?;
        self.stdin.flush()
    }

    fn wait_for(&mut self, token: &str) -> io::Result<String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "engine exited",
                ));
            }
            if line.starts_with(token) {
                return Ok(line.trim().to_string());
            }
        }
    }
}

impl Drop for Uci {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}
