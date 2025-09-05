#!/usr/bin/env python3
"""
Simple test script for dynamics module.
Run this to test physics changes without the full UI.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from geometry import create_mesh, compute_edges, build_hinge_graph
from dynamics import make_learned_energy_fn, make_hinge_energy_fn, compute_metric
from data_processing import load_default_data, calibrate_energy_curve


def test_dynamics():
    """Test basic dynamics functionality."""
    print("Testing dynamics...")
    
    # Create simple mesh
    vertices, faces = create_mesh(radius=1)
    edges = jnp.array(compute_edges(faces), dtype=jnp.int32)
    hinges = build_hinge_graph(faces)
    
    print(f"Mesh: {len(vertices)} vertices, {len(hinges)} hinges")
    
    # Load energy data if available
    data_path = Path("data/unit075_torque.txt")
    if data_path.exists():
        df = load_default_data(data_path, "auto")
        (_, _, _, _, _, theta_exp, E_exp) = calibrate_energy_curve(df, smoothing=0.5)
        learned_energy_fn = make_learned_energy_fn(theta_exp, E_exp)
        print("Loaded energy data")
    else:
        # Simple quadratic fallback
        learned_energy_fn = lambda theta: 0.5 * theta**2
        print("Using simple energy function")
    
    # Test energy function
    H = hinges.shape[0]
    hinge_state = jnp.zeros(H, dtype=jnp.int32)  # all neutral
    weights = jnp.ones(H, dtype=jnp.float32)
    
    hinge_energy_fn = make_hinge_energy_fn(faces, hinges, hinge_state, weights, learned_energy_fn)
    
    # Test on initial mesh
    X0 = jnp.array(vertices)
    energy = hinge_energy_fn(X0)
    print(f"Initial energy: {energy:.6f}")
    
    # Test gradient
    grad_fn = jax.grad(hinge_energy_fn)
    grad = grad_fn(X0)
    grad_norm = jnp.linalg.norm(grad)
    print(f"Gradient norm: {grad_norm:.6f}")
    
    print("✅ Dynamics test passed")


if __name__ == "__main__":
    test_dynamics() 