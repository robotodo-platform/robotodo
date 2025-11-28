

class InvalidReferenceError(ValueError):
    def __init__(self, ref: ...):
        super().__init__(f"Invalid reference type {type(ref)}: {ref}")