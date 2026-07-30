//! Example demonstrating optional Sentry error reporting.
//!
//! Run with: cargo run --features sentry --example sentry
//!
//! Set `SENTRY_DSN` to enable real reporting. Optional `SENTRY_ENVIRONMENT`
//! tags events (defaults to `development` so demo runs do not look like prod).
//!
//! This example only initializes the client and runs a normal neuromod step.
//! It does **not** send info-level probe messages — those create issue noise
//! (see instrumentation: errors at the edge only).

fn main() {
    // Guarded initialization - only when the "sentry" feature is enabled.
    #[cfg(feature = "sentry")]
    let _guard = {
        let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
        if !dsn.is_empty() {
            // Validate DSN before initialization
            match dsn.parse::<sentry::types::Dsn>() {
                Ok(parsed_dsn) => {
                    let environment = std::env::var("SENTRY_ENVIRONMENT")
                        .unwrap_or_else(|_| "development".into());
                    // sentry 0.49: ClientOptions is #[non_exhaustive]; use builder setters.
                    // release_name!() returns Option; builder setter wants Into<Cow<'static, str>>.
                    let mut options = sentry::ClientOptions::new().environment(environment.clone());
                    if let Some(release) = sentry::release_name!() {
                        options = options.release(release);
                    }
                    let guard = sentry::init((parsed_dsn, options));
                    println!(
                        "Sentry initialized for error monitoring (feature enabled, env={environment})"
                    );
                    Some(guard)
                }
                Err(e) => {
                    eprintln!("Invalid SENTRY_DSN format: {}", e);
                    println!("Sentry feature enabled but not reporting due to invalid DSN.");
                    None
                }
            }
        } else {
            println!("SENTRY_DSN not set; Sentry feature enabled but not reporting.");
            None
        }
    };

    #[cfg(not(feature = "sentry"))]
    {
        println!("Sentry feature not enabled. Running without error reporting.");
    }

    // Normal neuromod usage (always runs)
    use neuromod::{NeuroModulators, SpikingNetwork};
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();
    let _spikes = network.step(&stimuli, &modulators).expect("step failed");
    println!("neuromod step completed successfully.");
}
