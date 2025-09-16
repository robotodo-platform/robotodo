"""
TODO

"""


from typing import TypedDict


class Pose(TypedDict):
    p: ... # TODO xyz
    q: ... # TODO xyzw
    # angles: ... # TODO radian xyz

    # TODO __init__(p: ..., q: ..., angles: ...)
    # TODO facing(pose: "Pose")
