"""Independent continuum references in lattice coordinates; no solver imports.

Parameters below define the validation problem, rather than being inferred from
numerical solutions. H=N-1 is the plate distance; the periodic X length is N.
"""
from itertools import product
import numpy as np

NU = KAPPA = .02
AMPLITUDE = .1
CHANNEL_RE = .2


def exact(test, n, step):
    h = float(n-1)
    x = np.arange(n, dtype=float)[None, :]
    z = np.arange(n, dtype=float)[:, None]
    X, Z = x/n, z/h
    s = KAPPA*step/h**2
    theta = AMPLITUDE*np.exp(-s)*np.sin(2*np.pi*X)*np.sin(np.pi*Z)
    temperature = np.broadcast_to(1-Z, (n, n)).copy()
    velocity = np.zeros((3, n, n))
    force = np.zeros_like(velocity)
    stress = np.zeros((3, 3, n, n))
    source = np.zeros((n, n))
    lap_t = np.zeros_like(source)
    dt_t = np.zeros_like(source)
    adv_t = np.zeros_like(source)
    if test == 'V2':
        temperature += theta
        velocity[0] = KAPPA/h*np.sin(np.pi*Z)*np.cos(2*np.pi*s)
        dt_t = -KAPPA/h**2*theta
        lap_t = -((2*np.pi/n)**2+(np.pi/h)**2)*theta
        adv_t = velocity[0]*AMPLITUDE*np.exp(-s)*2*np.pi/n*np.cos(2*np.pi*X)*np.sin(np.pi*Z)
        source = KAPPA/h**2*(((2*np.pi*h/n)**2+np.pi**2-1)*theta)+adv_t
    if test == 'V3':
        s = NU*step/h**2
        amplitude = CHANNEL_RE*NU/h
        mod = 1+.25*np.sin(2*np.pi*s)
        velocity[0] = amplitude*mod*np.sin(np.pi*Z)
        dt_u = amplitude*.5*np.pi*np.cos(2*np.pi*s)*NU/h**2*np.sin(np.pi*Z)
        force[0] = dt_u+NU*(np.pi/h)**2*velocity[0]
        gradient = amplitude*mod*np.pi/h*np.cos(np.pi*Z)
        stress[0, 2] = stress[2, 0] = -NU*gradient
    # The exact Dirichlet data are set explicitly, avoiding sin(pi) roundoff.
    temperature[0] = 1
    temperature[-1] = 0
    velocity[:, [0, -1], :] = 0
    force[:, [0, -1], :] = 0
    source[[0, -1]] = 0
    return dict(T=temperature, u=velocity, a=force, stress=stress, source=source,
                lap_t=lap_t, dt_t=dt_t, adv_t=adv_t, time=s)


def stress_tensor(raw):
    p = np.empty((3, 3, *raw.shape[1:]))
    p[0, 0], p[1, 1], p[2, 2] = raw[4:7]
    p[0, 1] = p[1, 0] = raw[7]
    p[0, 2] = p[2, 0] = raw[8]
    p[1, 2] = p[2, 1] = raw[9]
    return p


def force_identity(previous, current, step, rr):
    """Independent Cartesian quadrature + direct Guo collision identity.

    Check only interior nodes: NEE is a different boundary operator. The
    comparison tests the local algebra, not a continuum truncation tolerance.
    """
    n = current.shape[-1]
    a_old = exact('V3', n, step-1)['a']
    a_new = exact('V3', n, step)['a']
    v = previous[1:4]
    u = v-a_old/2
    p = stress_tensor(previous)
    vv = np.einsum('izx,jzx->ijzx', v, v)
    c2 = p+np.einsum('izx,jzx->ijzx', a_old, a_old)/4
    a3 = np.empty((3, 3, 3, n, n))
    for i, j, k in product(range(3), repeat=3):
        a3[i, j, k] = (u[i]*u[j]*u[k]+u[i]*c2[j, k]+u[j]*c2[i, k]+u[k]*c2[i, j]
                       +(a_old[i]*u[j]*u[k]+a_old[j]*u[i]*u[k]+a_old[k]*u[i]*u[j])/2)
    rho = np.zeros((n, n))
    momentum = np.zeros((3, n, n))
    second = np.zeros((3, 3, n, n))
    eye = np.eye(3)
    for direction in product((-1, 0, 1), repeat=3):
        c = np.array(direction)
        weight = np.prod(np.where(c == 0, 2/3, 1/6))
        h2 = np.outer(c, c)-eye/3
        value = 1+3*np.einsum('i,izx->zx', c, v)+4.5*np.einsum('ij,ijzx->zx', h2, vv+p)
        if rr == 3:
            h3 = np.empty((3, 3, 3))
            for i, j, k in product(range(3), repeat=3):
                h3[i, j, k] = c[i]*c[j]*c[k]-(c[i]*eye[j, k]+c[j]*eye[i, k]+c[k]*eye[i, j])/3
            value += 4.5*np.einsum('ijk,ijkzx->zx', h3, a3)
        f = weight*previous[0]*value
        # Roll in Z is harmless on discarded plate rows; X is truly periodic.
        pulled = np.roll(f, shift=(int(c[2]), int(c[0])), axis=(0, 1))
        rho += pulled
        momentum += c[:, None, None]*pulled
        second += h2[:, :, None, None]*pulled
    q = momentum/rho
    pre_a2 = second/rho
    physical = q+a_new/2
    omega = 1/(.5+3*NU)
    target_a2 = ((1-omega)*pre_a2+omega*np.einsum('izx,jzx->ijzx', physical, physical)
                 +(1-omega/2)*(np.einsum('izx,jzx->ijzx', physical, a_new)
                              +np.einsum('izx,jzx->ijzx', a_new, physical)))
    actual_a2 = stress_tensor(current)+np.einsum('izx,jzx->ijzx', current[1:4], current[1:4])
    interior = (..., slice(1, -1), slice(None))
    residual = max(float(np.max(np.abs((current[0]-rho)[1:-1]))),
                   float(np.max(np.abs((current[1:4]-q-a_new)[interior]))),
                   float(np.max(np.abs((actual_a2-target_a2)[interior]))))
    # Conservative forward-error budget for the independent polynomial sum,
    # divisions and collision identity. This is NOT an NS accuracy tolerance.
    eps = np.finfo(float).eps
    gamma = 65536*eps/(1-65536*eps)
    condition = (max(1., float(previous[0].max()))/min(1., float(rho[1:-1].min())))**2
    scale = (1+np.max(np.abs(v))+np.max(np.abs(p))+np.max(np.abs(a_new)))**3
    return residual, float(gamma*condition*scale)
