# Warning audit

The remaining warnings are third-party `joblib`/NumPy deprecation warnings emitted while loading persisted pipelines under NumPy 2.5. No project-code warnings were identified. They do not prevent loading or prediction; they should be resolved by rebuilding artifacts after the dependency stack stabilizes.
