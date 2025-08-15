//! Basic usage example of the twentyone-core library.

use twentyone_core::{Action, Env};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Twenty-One Core Library Demo");
    println!("============================");
    
    // Create environment
    let mut env = Env::new(42);
    println!("Created environment with {} hearts each", env.hearts(0));
    
    // Play a single round
    println!("\nStarting round {}...", env.round());
    env.start_new_round()?;
    
    // Simple strategy: draw if total < 17, otherwise stand
    while {
        let player = env.current_player();
        let obs = env.observation(player);
        
        println!("Player {}: total={}, opp_face_up={}", 
                 player, obs.self_total, obs.opp_face_up);
        
        let action = if obs.self_total < 17 { 
            Action::Draw 
        } else { 
            Action::Stand 
        };
        
        println!("Player {} chooses: {:?}", player, action);
        
        let result = env.step(action)?;
        
        if result.round_over {
            println!("Round over!");
            if let Some(outcome) = result.outcome {
                match outcome.winner {
                    Some(winner) => println!("Winner: Player {}", winner),
                    None => println!("Tie!"),
                }
            }
            println!("Hearts: P0={}, P1={}", env.hearts(0), env.hearts(1));
            false // Exit loop
        } else {
            true // Continue loop
        }
    } {}
    
    println!("\n✅ Demo completed!");
    Ok(())
}