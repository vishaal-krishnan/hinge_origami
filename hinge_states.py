import numpy as np
import jax
import jax.numpy as jnp
import re


def create_probabilistic_states(H, p_neg, p_zero, p_pos, state_seed):
    """
    Create hinge states by sampling from probabilities.
    
    Args:
        H: Number of hinges
        p_neg, p_zero, p_pos: Probabilities for states -1, 0, +1
        state_seed: Random seed
    
    Returns:
        jnp.array: Hinge states of shape (H,)
    """
    # Normalize probabilities
    p_sum = p_neg + p_zero + p_pos
    if p_sum == 0:
        p_neg = p_zero = p_pos = 1/3
    else:
        p_neg, p_zero, p_pos = np.array([p_neg, p_zero, p_pos]) / p_sum
    
    state_vals = jnp.array([-1, 0, 1])
    probs = jnp.array([p_neg, p_zero, p_pos], dtype=jnp.float32)
    key_states = jax.random.PRNGKey(int(state_seed))
    
    return jax.random.choice(key_states, state_vals, (H,), p=probs)


def create_manual_states(H, default_all, manual_text):
    """
    Create hinge states by manual assignment.
    
    Args:
        H: Number of hinges
        default_all: Default value for all hinges
        manual_text: Optional custom list as text
    
    Returns:
        tuple: (hinge_state, error_message)
               hinge_state is jnp.array of shape (H,) or None if error
               error_message is string or None if success
    """
    if manual_text.strip():
        # Parse list of ints from text (split on comma/space/newline)
        tokens = [t for t in re.split(r"[,\s]+", manual_text.strip()) if t]
        try:
            vals = np.array([int(t) for t in tokens], dtype=int)
        except ValueError:
            return None, "Manual list contains non-integer entries. Use only -1, 0, or 1."
        
        if np.any(~np.isin(vals, [-1, 0, 1])):
            return None, "Manual list may only contain -1, 0, or 1."
        
        if len(vals) != H:
            return None, f"Manual list length ({len(vals)}) must match number of hinges ({H})."
        
        return jnp.array(vals, dtype=jnp.int32), None
    else:
        # Apply one value to all hinges
        return jnp.full((H,), int(default_all), dtype=jnp.int32), None


def validate_probabilities(p_neg, p_zero, p_pos):
    """Validate and normalize probabilities."""
    p_sum = p_neg + p_zero + p_pos
    if p_sum == 0:
        return 1/3, 1/3, 1/3, "Sum of probabilities is 0; normalizing to equal thirds."
    else:
        p_neg, p_zero, p_pos = np.array([p_neg, p_zero, p_pos]) / p_sum
        return p_neg, p_zero, p_pos, None 