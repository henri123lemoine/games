use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, BufRead, Write};
use twentyone::env::{Action, Env, Observation, RoundOutcome};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    New { seed: u64 },
    StartRound,
    Observation { player: usize },
    Step { action: Act },
    Hearts,
    Round,
    CurrentPlayer,
    Quit,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Act {
    Draw,
    Stand,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response<T> {
    Ok { data: T },
    Err { error: String },
}

#[derive(Serialize, Deserialize, Debug)]
struct ObsOut {
    observation: Observation,
}

#[derive(Serialize, Deserialize, Debug)]
struct StepOut {
    step: StepResultOut,
}

#[derive(Serialize, Deserialize, Debug)]
struct StepResultOut {
    round_over: bool,
    game_over: bool,
    outcome: Option<RoundOutcome>,
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut env: Option<Env> = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let cmd: Result<Command, _> = serde_json::from_str(&line);
        match cmd {
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    serde_json::to_string(&Response::<()>::Err {
                        error: e.to_string()
                    })
                    .unwrap()
                );
                let _ = stdout.flush();
                continue;
            }
            Ok(c) => {
                let resp = handle(&mut env, c);
                let _ = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap());
                let _ = stdout.flush();
            }
        }
    }
}

fn handle(env: &mut Option<Env>, cmd: Command) -> Response<serde_json::Value> {
    match cmd {
        Command::New { seed } => {
            *env = Some(Env::new(seed));
            ok(json!({"ok": true}))
        }
        Command::StartRound => match env {
            Some(e) => match e.start_new_round() {
                Ok(_) => ok(json!({"ok": true})),
                Err(err) => err_resp(err.to_string()),
            },
            None => err_resp("env_not_initialized".into()),
        },
        Command::Observation { player } => match env {
            Some(e) => ok(json!({"observation": e.observation(player)})),
            None => err_resp("env_not_initialized".into()),
        },
        Command::Step { action } => match env {
            Some(e) => {
                let act = match action {
                    Act::Draw => Action::Draw,
                    Act::Stand => Action::Stand,
                };
                match e.step(act) {
                    Ok(sr) => ok(json!({
                        "step": {
                            "round_over": sr.round_over,
                            "game_over": sr.game_over,
                            "outcome": sr.outcome,
                        }
                    })),
                    Err(err) => err_resp(err.to_string()),
                }
            }
            None => err_resp("env_not_initialized".into()),
        },
        Command::Hearts => match env {
            Some(e) => ok(json!({"p0": e.hearts(0), "p1": e.hearts(1)})),
            None => err_resp("env_not_initialized".into()),
        },
        Command::Round => match env {
            Some(e) => ok(json!({"round": e.round()})),
            None => err_resp("env_not_initialized".into()),
        },
        Command::CurrentPlayer => match env {
            Some(e) => ok(json!({"current_player": e.current_player()})),
            None => err_resp("env_not_initialized".into()),
        },
        Command::Quit => ok(json!({"bye": true})),
    }
}

fn ok<T: Serialize>(data: T) -> Response<serde_json::Value> {
    Response::Ok {
        data: serde_json::to_value(data).unwrap(),
    }
}

fn err_resp<T>(msg: String) -> Response<T> {
    Response::Err { error: msg }
}
