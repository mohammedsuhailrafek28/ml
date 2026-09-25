"""Compatibility entry point backed by the production inference registry."""


def predict(disease, values):
    from src.api.services.model_registry import registry

    return registry.predict(disease, values)
