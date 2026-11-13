"""Designed single-agent door games for the tacking tests (theory.md S4).

DoorTrap  root: 'safe' pays 1.0, 'junk1', 'junk2' pay 0, 'door' leads to key k' (after a fair coin c that the
          player observes: two copies of k', so the door is a two-key sub-tree): 'good' pays 2.0, 'bad1', 'bad2'
          pay 0. The optimum takes door -> good (V_opt 2.0). A door closed at the root AND closed behind
          (pi(door) = pi(good) = 1%) is the trap: Q^pi(door) = 2 pi(good) < 1, so the south field pushes the door
          further shut and the keys behind it get occupancy ~ pi(door) (Agarwal et al.'s mismatch).
FalseDoor root: 'safe' pays 1.1; 'door' leads to k' where a hidden coin c (NOT observed) decides which of
          'left' / 'right' pays 2.0 (the other 0): Q(door) = 1.0 < 1.1, but the clairvoyant (who sees c) values the
          door at 2.0. North keeps flagging a door that is truly worse: the bias case.
"""


class DoorTrap:
    name = 'DoorTrap'
    players = 1
    merge = True

    def deals(self):
        return [(0.5, 0), (0.5, 1)]

    def root(self, d):
        return ('root',)

    def terminal(self, d, s):
        return s[0] == 'end', (s[1] if s[0] == 'end' else 0.0)

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        if s[0] == 'root':
            return ['safe', 'junk1', 'junk2', 'door']
        return ['good', 'bad1', 'bad2']

    def step(self, d, s, a):
        if s[0] == 'root':
            if a == 'door':
                return ('inside',), 0.0
            return ('end', 1.0 if a == 'safe' else 0.0), 0.0
        return ('end', 2.0 if a == 'good' else 0.0), 0.0

    def key(self, d, s, i):
        return s if s[0] == 'root' else (s, d)          # the coin is observed behind the door


class FalseDoor:
    name = 'FalseDoor'
    players = 1
    merge = True

    def deals(self):
        return [(0.5, 0), (0.5, 1)]

    def root(self, d):
        return ('root',)

    def terminal(self, d, s):
        return s[0] == 'end', (s[1] if s[0] == 'end' else 0.0)

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        if s[0] == 'root':
            return ['safe', 'junk', 'door']
        return ['left', 'right']

    def step(self, d, s, a):
        if s[0] == 'root':
            if a == 'door':
                return ('inside',), 0.0
            return ('end', 1.1 if a == 'safe' else 0.0), 0.0
        return ('end', 2.0 if (a == 'left') == (d == 0) else 0.0), 0.0

    def key(self, d, s, i):
        return s                                        # the coin is hidden


class Sailboat:
    """Each field alone fails, the tack succeeds. root: 'safe' 1.0, 'junk' 0, 'door' (closed) -> inside, where
    'good' pays 2.0 (closed), 'bad' pays 0, and 'gamble' leads to a bet on a hidden coin (heads / tails: 2.4 if
    right, -0.4 if wrong; true EV 1.0, clairvoyant 2.4 -- the clairvoyance needs the LATER choice).  South alone: the door stays locked (gap 1.0). North alone: through the door,
    then the gamble (V 1.0, gap 1.0). Tack: north opens the door, south picks 'good' behind it; the gate holds the
    gamble near eps (bias <= eps x 1.0)."""
    name = 'Sailboat'
    players = 1
    merge = True

    def deals(self):
        return [(0.5, 0), (0.5, 1)]

    def root(self, d):
        return ('root',)

    def terminal(self, d, s):
        return s[0] == 'end', (s[1] if s[0] == 'end' else 0.0)

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        if s[0] == 'root':
            return ['safe', 'junk', 'door']
        if s[0] == 'inside':
            return ['good', 'bad', 'gamble']
        return ['heads', 'tails']

    def step(self, d, s, a):
        if s[0] == 'root':
            if a == 'door':
                return ('inside',), 0.0
            return ('end', 1.0 if a == 'safe' else 0.0), 0.0
        if s[0] == 'bet':
            return ('end', 2.4 if (a == 'heads') == (d == 0) else -0.4), 0.0
        if a == 'good':
            return ('end', 2.0), 0.0
        if a == 'bad':
            return ('end', 0.0), 0.0
        return ('bet',), 0.0

    def key(self, d, s, i):
        return s


class WrongDoor:
    """North ranks the closed doors wrongly. root: 'safe' 1.0, 'door' (closed) -> 'good' 2.0 (closed) / 'bad' 0;
    'false' (closed) -> a hidden-coin bet paying 2.2 / -0.6 (true EV 0.8, clairvoyant 2.2). The north argmax at the
    root is the false door, so the D23a rule (fire on the argmax only) never opens the true door."""
    name = 'WrongDoor'
    players = 1
    merge = True

    def deals(self):
        return [(0.5, 0), (0.5, 1)]

    def root(self, d):
        return ('root',)

    def terminal(self, d, s):
        return s[0] == 'end', (s[1] if s[0] == 'end' else 0.0)

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        if s[0] == 'root':
            return ['safe', 'junk', 'door', 'false']
        if s[0] == 'in_door':
            return ['good', 'bad']
        return ['heads', 'tails']

    def step(self, d, s, a):
        if s[0] == 'root':
            if a == 'door':
                return ('in_door',), 0.0
            if a == 'false':
                return ('in_false',), 0.0
            return ('end', 1.0 if a == 'safe' else 0.0), 0.0
        if s[0] == 'in_door':
            return ('end', 2.0 if a == 'good' else 0.0), 0.0
        return ('end', 2.2 if (a == 'heads') == (d == 0) else -0.6), 0.0

    def key(self, d, s, i):
        return s
