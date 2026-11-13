"""Game registry: build(name) -> polar.plex.Game."""
from ..plex import Game


def build(name):
    from . import door, ladder, matrix, poker
    if name in matrix.MATS:
        return Game(matrix.Matrix(name), name)
    procs = dict(Kuhn=poker.Kuhn, Leduc=poker.Leduc, L0=ladder.L0, L1=ladder.L1, L2=ladder.L2, R2=ladder.R2,
                 DoorTrap=door.DoorTrap, FalseDoor=door.FalseDoor,
                 Sailboat=door.Sailboat, WrongDoor=door.WrongDoor)
    return Game(procs[name](), name)


ALL = ['MP', 'RPS', 'MPdup', 'MPtri', 'Skew', 'Kuhn', 'Leduc', 'L0', 'L1', 'L2', 'R2']
