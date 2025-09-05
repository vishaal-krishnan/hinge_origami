import io
import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from pathlib import Path


def read_dataframe(uploaded, delim):
    """Read uploaded file into DataFrame."""
    if uploaded is None: 
        return None
    
    data = uploaded.read()
    if delim == "auto":
        try:
            df = pd.read_csv(io.BytesIO(data), header=None, sep=r"\s+")
        except:
            df = pd.read_csv(io.BytesIO(data), header=None)
    elif delim == "space":
        df = pd.read_csv(io.BytesIO(data), header=None, sep=r"\s+")
    elif delim == "comma":
        df = pd.read_csv(io.BytesIO(data), header=None)
    else:  # tab
        df = pd.read_csv(io.BytesIO(data), header=None, sep="\t")
    
    df.columns = ["Angle", "Torque"]
    return df


def load_default_data(default_path, delim):
    """Load default torque data file."""
    if not default_path.exists():
        raise FileNotFoundError(f"Default file not found at: {default_path}")
    
    # Try to honor the delimiter choice; default is whitespace
    if delim in ("auto", "space"):
        df = pd.read_csv(default_path, header=None, sep=r"\s+")
    elif delim == "comma":
        df = pd.read_csv(default_path, header=None)
    else:  # tab
        df = pd.read_csv(default_path, header=None, sep="\t")
    
    df.columns = ["Angle", "Torque"]
    return df


def calibrate_energy_curve(df, smoothing=0.5):
    """
    Calibrate spline and energy curve from torque-angle data.
    
    Returns:
        tuple: (angle_smooth, torque_smooth, slope_smooth, energy_smooth, 
                alpha_ref, theta_exp, E_exp)
    """
    # Sort by angle and convert to radians
    sort_idx = np.argsort(df['Angle'].values)
    angles = df['Angle'].values[sort_idx] * np.pi / 180.0
    torques = df['Torque'].values[sort_idx]
    
    # Fit spline
    spline = UnivariateSpline(angles, torques, s=float(smoothing))
    
    # Generate smooth curves
    angle_smooth = np.linspace(angles.min(), angles.max(), 500)
    torque_smooth = spline(angle_smooth)
    slope_smooth = spline.derivative()(angle_smooth)
    
    # Find neutral angle (zero torque) to recenter about 0
    roots = spline.roots()
    if len(roots) > 0:
        alpha_ref = float(roots[np.argmin(np.abs(roots))])  # root closest to 0
    else:
        # fallback: where |torque| is minimal
        alpha_ref = float(angle_smooth[np.argmin(np.abs(torque_smooth))])
    
    # Integrate energy
    energy_smooth = cumulative_trapezoid(torque_smooth, angle_smooth, initial=0.0)
    energy_smooth -= energy_smooth.min()
    
    # Convert to JAX arrays (ensure float32 for JAX)
    theta_exp = jnp.array(angle_smooth, dtype=jnp.float32)
    E_exp = jnp.array(energy_smooth, dtype=jnp.float32)
    
    return (angle_smooth, torque_smooth, slope_smooth, energy_smooth, 
            alpha_ref, theta_exp, E_exp) 