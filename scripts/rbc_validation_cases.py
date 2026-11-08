"""Declarative pre-production matrix. Importing this module never launches work."""
from dataclasses import dataclass, asdict

GRIDS = {'V1': (16, 32, 64, 128), 'V2': (32, 64, 128),
         'V3': (32, 64, 128), 'V4': (64, 128), 'V5': (128, 256)}
FORMULA_VERSION = 1


@dataclass(frozen=True)
class Job:
    test: str
    n: int
    rr: int
    precision: str
    case: int
    steps: int
    transient_steps: int
    sample: int
    checkpoint_every: int
    formula_version: int = FORMULA_VERSION

    @property
    def run_id(self):
        return f'validation{self.formula_version}_{self.test.lower()}_n{self.n:03d}_rr{self.rr}_{self.precision}'

    def description(self):
        return dict(asdict(self), run_id=self.run_id, h_z=1/(self.n-1),
                    h_x=1/(self.n-1), periodic_coordinate_spacing=1/self.n,
                    aspect_ratio=self.n/(self.n-1))


def jobs(selected, rr='3', checkpoint_every=None):
    for test in GRIDS:
        if test not in selected:
            continue
        orders = (2, 3) if test == 'V3' and rr == 'both' else (int(rr),) if test == 'V3' else (3,)
        for n in GRIDS[test]:
            h = n-1
            for order in orders:
                if test in ('V1', 'V2', 'V3'):
                    steps = h*h if test == 'V1' else 5*h*h
                    transient, sample = 0, max(1, steps//40)
                    stamp = max(1, steps//4)
                    precision, case = 'double', 1
                else:
                    case = 1 if test == 'V4' else 2
                    transient = 100*20*h
                    steps = (250 if test == 'V4' else 300)*20*h
                    sample, stamp = 100, 10*20*h
                    precision = 'single'
                yield Job(test, n, order, precision, case, steps, transient, sample,
                          checkpoint_every or stamp)
