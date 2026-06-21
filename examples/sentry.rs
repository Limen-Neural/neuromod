//! Example demonstrating optional Sentry error reporting.
//!
//! Run with: cargo run --features sentry --example sentry
//!
//! Set SENTRY_DSN environment variable to enable real reporting.

fn main() {
    // Guarded initialization - only when the "sentry" feature is enabled.
    #[cfg(feature = "sentry")]
    let _guard = {
        let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
        if !dsn.is_empty() {
            let guard = sentry::init((
                dsn,
                sentry::ClientOptions {
                    release: sentry::release_name!(),
                    ..Default::default()
                },
            ));
            println!("Sentry initialized for error monitoring (feature enabled)");
            Some(guard)
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
