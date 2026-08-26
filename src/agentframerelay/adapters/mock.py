from ..runtime import RuntimeAdapter, RuntimeResult

class MockAdapter(RuntimeAdapter):
    name = "mock"

    @classmethod
    def build(cls, spec):
        return {"spec": spec}

    @classmethod
    def run(cls, native_agent, input, **kwargs):
        spec = native_agent["spec"]
        return RuntimeResult(
            output={"message": "Mock runtime executed.", "input": input,
                    "agent": spec.name, "strategy": spec.strategy},
            runtime=cls.name,
        )
