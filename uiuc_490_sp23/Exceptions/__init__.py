"""All package specific exceptions should go here"""


class DimensionMismatch(ValueError):
    """Something in your problem formulation is borked"""


class BadLineSearchSpec(ValueError):
    """Something is dorked in your Line search Specs"""
