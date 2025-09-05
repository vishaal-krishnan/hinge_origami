import jax
import jax.numpy as jnp
import optax
from geometry import face_normals, signed_dihedral_angles


@jax.jit
def compute_metric(X, edges):
    """Compute edge length metric."""
    return jnp.sum((X[edges[:, 0]] - X[edges[:, 1]])**2, axis=-1)


@jax.jit
def isometry_loss(X, edges, initial_metric):
    """Compute isometry preservation loss."""
    return jnp.mean((compute_metric(X, edges) - initial_metric) ** 2)


@jax.jit
def collision_penalty(X, r=0.5):
    """Compute collision penalty between vertices."""
    D2 = jnp.sum((X[:, None] - X[None])**2, axis=-1)
    mask = (D2 > 1e-6) & (D2 < r**2)
    return jnp.sum(jnp.where(mask, (r**2 - D2)**2, 0.0)) / 2


collision_force = jax.grad(collision_penalty)


def project_constraints(X, V, edges, metric, w_col=1.0, steps=20, lr=0.01):
    """Project velocity to satisfy both isometry and collision constraints simultaneously."""
    opt = optax.adam(lr)
    state = opt.init(V)
    
    def loss_fn(V):
        X_new = X + V
        # Isometry constraint loss
        iso_loss = jnp.sum((compute_metric(X_new, edges) - metric)**2)
        # Collision constraint loss  
        col_loss = collision_penalty(X_new)
        # Combined objective
        return iso_loss + w_col * col_loss
    
    def step(carry, _):
        V, opt_state = carry
        loss, grad = jax.value_and_grad(loss_fn)(V)
        updates, opt_state = opt.update(grad, opt_state)
        return (optax.apply_updates(V, updates), opt_state), loss
    
    (V_final, _), losses = jax.lax.scan(step, (V, state), None, length=steps)
    return V_final, losses


def make_learned_energy_fn(theta_exp, E_exp):
    """Create learned energy function from experimental data."""
    def learned_energy_plus(theta):
        theta_clamped = jnp.clip(theta, theta_exp[0], theta_exp[-1])
        return jnp.interp(theta_clamped, theta_exp, E_exp)
    return learned_energy_plus


def make_hinge_energy_fn(faces, hinges, state, weights, learned_energy_fn):
    """
    Create hinge energy function with state-dependent behavior:
    +1 -> E(θ),  -1 -> E(-θ),  0 -> 0.5*(E(θ)+E(-θ))
    """
    faces = jnp.asarray(faces)
    hinges = jnp.asarray(hinges)
    state = jnp.asarray(state)        # shape (H,)
    weights = jnp.asarray(weights)    # shape (H,)

    def energy_fn(X):
        theta = signed_dihedral_angles(X, faces, hinges)  # (H,)

        E_plus  = learned_energy_fn(theta)              # E(θ)
        E_minus = learned_energy_fn(-theta)             # E(-θ)
        E_zero  = 0.5 * (E_plus + E_minus)              # average

        E_state = jnp.where(state ==  1, E_plus,
                   jnp.where(state == -1, E_minus, E_zero))

        return jnp.sum(weights * E_state)

    return energy_fn


def diffusion_step_euler(X, key, edges, faces, metric,
                        dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                        hinge_grad):
    """Single step using Euler method (1st order)."""
    # Drift from gradient of total hinge energy
    drift = -theta_gain * hinge_grad(X)             # (N,3)
    
    # Noise
    noise = sigma * jax.random.normal(key, X.shape)
    V = drift * dt + noise * jnp.sqrt(dt)
    
    # Project to satisfy both isometry and collision constraints
    V_proj, constraint_losses = project_constraints(X, V, edges, metric, w_col, proj_steps, proj_lr)
    
    return X + V_proj


def diffusion_step_heun(X, key, edges, faces, metric,
                       dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                       hinge_grad):
    """Single step using Heun's method (2nd order) - more stable for varying dt."""
    key1, key2 = jax.random.split(key)
    
    # Same noise for both predictor and corrector steps
    noise = sigma * jax.random.normal(key1, X.shape)
    
    # Predictor step (Euler)
    drift1 = -theta_gain * hinge_grad(X)
    V1 = drift1 * dt + noise * jnp.sqrt(dt)
    V1_proj, _ = project_constraints(X, V1, edges, metric, w_col, proj_steps, proj_lr)
    X_pred = X + V1_proj
    
    # Corrector step
    drift2 = -theta_gain * hinge_grad(X_pred)
    V2 = 0.5 * (drift1 + drift2) * dt + noise * jnp.sqrt(dt)
    V2_proj, constraint_losses = project_constraints(X, V2, edges, metric, w_col, proj_steps, proj_lr)
    
    return X + V2_proj


def diffusion_step(X, key, edges, faces, metric,
                   dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                   hinge_grad, method='heun'):
    """Single step of the diffusion dynamics with choice of integration method."""
    if method == 'euler':
        return diffusion_step_euler(X, key, edges, faces, metric,
                                   dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                                   hinge_grad)
    elif method == 'heun':
        return diffusion_step_heun(X, key, edges, faces, metric,
                                  dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                                  hinge_grad)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def simulate(key, vertices, faces, edges, hinges, hinge_state,
             steps, dt, sigma, theta_gain, w_col,
             proj_steps, proj_lr,
             hinge_grad, method='heun'):
    """Run the full simulation."""
    X0 = jnp.array(vertices)
    metric0 = compute_metric(X0, edges)

    def step(X, key):
        key, sub = jax.random.split(key)
        X_new = diffusion_step(X, sub, edges, faces, metric0,
                               dt, theta_gain, sigma, w_col, proj_steps, proj_lr,
                               hinge_grad, method)
        return X_new, X_new

    keys = jax.random.split(key, steps)
    _, traj = jax.lax.scan(step, X0, keys)
    return jnp.concatenate([X0[None], traj], axis=0) 