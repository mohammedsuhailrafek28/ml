def validate_input(values, features):
    missing=[f for f in features if f not in values or values[f] is None]
    if missing: raise ValueError(f"Missing fields: {', '.join(missing)}")
    return True
