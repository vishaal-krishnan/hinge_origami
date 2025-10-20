import streamlit as st
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Import our modules
from geometry import create_mesh, compute_edges, build_hinge_graph, compute_dihedrals_robust
from dynamics import make_learned_energy_fn, make_hinge_energy_fn, simulate
from visualization import plot_torque_stiffness_energy, create_plotly_animated_trajectory, plot_hinge_angles
from data_processing import read_dataframe, load_default_data, calibrate_energy_curve
from hinge_states import create_probabilistic_states, create_manual_states, validate_probabilities

# Path to bundled default torque file
DEFAULT_TORQUE_PATH = Path(__file__).parent / "data" / "unit075_torque.txt"

# -----------------------
# UI Setup
# -----------------------
st.set_page_config(page_title="Origami Hinge Simulator", layout="wide")
st.title("Origami Hinge Simulator")

# Sidebar controls for file upload
st.sidebar.header("1) Upload torque–angle data")
uploaded = st.sidebar.file_uploader("File (.txt/.csv): two columns (Angle, Torque)", type=["txt", "csv"])

delim = st.sidebar.selectbox("Delimiter", ["auto", "space", "comma", "tab"], index=0)
smoothing = st.sidebar.slider("Spline smoothing (s)", 0.0, 5.0, 0.5, 0.1)

# Sidebar controls for hinge states
st.sidebar.header("2) Hinge states")

hinge_mode = st.sidebar.radio(
    "How do you want to set hinge states?",
    ["Sample by probabilities", "Manual assignment"],
    index=0
)

if hinge_mode == "Sample by probabilities":
    p_neg = st.sidebar.slider("P(state = -1)", 0.0, 1.0, 0.33, 0.01)
    p_zero = st.sidebar.slider("P(state = 0)", 0.0, 1.0, 0.34, 0.01)
    p_pos = st.sidebar.slider("P(state = +1)", 0.0, 1.0, 0.33, 0.01)
    
    # Validate probabilities
    p_neg, p_zero, p_pos, warning = validate_probabilities(p_neg, p_zero, p_pos)
    if warning:
        st.sidebar.warning(warning)

    state_seed = st.sidebar.number_input("Seed (hinge-state sampling)", value=0, step=1)
else:
    # Manual assignment UI
    st.sidebar.markdown("**Manual assignment**")
    default_all = st.sidebar.selectbox("Set ALL hinges to:", [-1, 0, 1], index=1)
    manual_text = st.sidebar.text_area(
        "Optional: paste custom list (-1 0 1 ...), separated by space/comma/newline",
        value=""
    )
    state_seed = None  # not used in manual mode

# Sidebar controls for simulation
st.sidebar.header("3) Simulation")
steps = st.sidebar.number_input("Steps", 100, 10000, 1000, 100)
dt = st.sidebar.number_input("dt", 1e-5, 1.0, 0.01, 0.01, format="%.5f")
sigma = st.sidebar.number_input("Noise σ", 0.0, 0.1, 1e-4, 1e-4, format="%.6f")
theta_gain = st.sidebar.number_input("Theta gain (force scale)", 0.0, 10.0, 1.0, 0.1)
w_col = st.sidebar.number_input("Collision weight", 0.0, 100.0, 10.0, 1.0)
proj_steps = st.sidebar.number_input("Isometry proj steps", 0, 1000, 50, 10)
proj_lr = st.sidebar.number_input("Isometry proj lr", 1e-5, 1e-1, 1e-2, 1e-3, format="%.5f")
radius = st.sidebar.number_input("Hex radius (tiles)", 0, 3, 0, 1)

integration_method = st.sidebar.selectbox("Integration method", 
                                                    ["rk4", "heun", "euler", "adaptive"], 
                                                    index=0,
                                                    help="RK4: 4th order accuracy (recommended)\nHeun: 2nd order\nEuler: 1st order\nAdaptive: Error control")

sim_seed = st.sidebar.number_input("Seed (simulation noise)", value=1, step=1)
run_btn = st.sidebar.button("Run Simulation")

# Configuration analysis
st.sidebar.header("4) Configuration Analysis")
analyze_btn = st.sidebar.button("Run 100 Simulations & Analyze Configs", help="Run 100 simulations with different seeds to identify unique final configurations")
angle_tolerance = st.sidebar.number_input("Angle tolerance (degrees)", 1.0, 10.0, 2.0, 0.5, help="Tolerance for grouping similar configurations")

# -----------------------
# Data Loading and Processing
# -----------------------

# Load data (uploaded or default)
if uploaded:
    df = read_dataframe(uploaded, delim)
    st.success("File uploaded successfully!")
else:
    st.info("Using default torque.txt")
    try:
        df = load_default_data(DEFAULT_TORQUE_PATH, delim)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# Preview data
st.write("Preview:", df.head())

# Calibrate spline and energy
(angle_smooth, torque_smooth, slope_smooth, energy_smooth, 
 alpha_ref, theta_exp, E_exp) = calibrate_energy_curve(df, smoothing)

# Create learned energy function
learned_energy_fn = make_learned_energy_fn(theta_exp, E_exp)

# Display calibration plots
st.subheader("Calibration Plots")
st.plotly_chart(
    plot_torque_stiffness_energy(angle_smooth, torque_smooth, slope_smooth, energy_smooth),
    use_container_width=True
)

# -----------------------
# Mesh and Hinge Setup
# -----------------------

# Build mesh, edges, hinges
vertices, faces = create_mesh(radius=int(radius))
edges = jnp.array(compute_edges(faces), dtype=jnp.int32)
hinges = build_hinge_graph(faces)

# Create hinge states
H = int(hinges.shape[0])  # number of hinges

if hinge_mode == "Sample by probabilities":
    hinge_state = create_probabilistic_states(H, p_neg, p_zero, p_pos, state_seed)
else:
    # Manual assignment
    hinge_state, error_msg = create_manual_states(H, default_all, manual_text)
    if error_msg:
        st.error(error_msg)
        st.stop()

# Show preview
st.caption(f"Hinges: {H} — mode: {hinge_mode}")

# -----------------------
# Simulation
# -----------------------

if run_btn:
    st.subheader("Simulation Running…")
    key = jax.random.PRNGKey(int(sim_seed))

    # Per-hinge weights (uniform for now)
    weights = jnp.ones(hinges.shape[0], dtype=jnp.float32)

    # Build total energy(X) with the current hinge_state and precompute its grad
    hinge_energy_fn = make_hinge_energy_fn(faces, hinges, hinge_state, weights, learned_energy_fn)
    hinge_grad = jax.grad(hinge_energy_fn)

    traj = simulate(
        key, vertices, faces, edges, hinges, hinge_state,
        steps=int(steps), dt=float(dt), sigma=float(sigma),
        theta_gain=float(theta_gain), w_col=float(w_col),
        proj_steps=int(proj_steps), proj_lr=float(proj_lr),
        hinge_grad=hinge_grad,
        method=integration_method
    )
    st.success("Done.")

    # Animation
    st.subheader("Mesh Animation")
    anim_fig = create_plotly_animated_trajectory(
        traj, faces, edges, frame_stride=max(1, int(steps//50))
    )
    st.plotly_chart(anim_fig, use_container_width=True)

    # Robust dihedrals and hinge plot
    st.subheader("Hinge Angles (0–360°)")
    X_final = np.asarray(traj[-1])
    angles_0_2pi, faces_oriented, hinges_ordered = compute_dihedrals_robust(X_final, faces)
    hinge_fig = plot_hinge_angles(X_final, faces_oriented, hinges_ordered, angles_0_2pi)
    st.plotly_chart(hinge_fig, use_container_width=True)

    # Download HTMLs
    st.subheader("Downloads")
    anim_html = anim_fig.to_html(full_html=True, include_plotlyjs='cdn')
    hinge_html = hinge_fig.to_html(full_html=True, include_plotlyjs='cdn')
    st.download_button("Download Animation (HTML)", data=anim_html, file_name="trajectory.html", mime="text/html")
    st.download_button("Download Hinge Angles (HTML)", data=hinge_html, file_name="hinge_angles.html", mime="text/html")

else:
    st.info("Upload your torque–angle file to begin.")

# -----------------------
# Configuration Analysis (100 simulations)
# -----------------------

def normalize_angles_to_degrees(angles):
    """Convert angles to degrees in [0, 360) range."""
    return np.mod(np.degrees(angles), 360.0)

def are_cyclic_permutations(config1, config2, tolerance_deg=2.0):
    """Check if two configurations are cyclic permutations of each other."""
    n = len(config1)
    if len(config2) != n:
        return False
    
    # Try all possible rotations
    for shift in range(n):
        rotated = np.roll(config2, shift)
        if np.allclose(config1, rotated, atol=tolerance_deg):
            return True
    return False

def group_configurations(all_configs, tolerance_deg=2.0):
    """
    Group configurations, removing cyclic permutation duplicates.
    
    Returns:
        List of tuples: (representative_config, list_of_indices)
    """
    unique_groups = []
    
    for i, config in enumerate(all_configs):
        # Check if this config matches any existing group
        matched = False
        for group_config, indices in unique_groups:
            if are_cyclic_permutations(config, group_config, tolerance_deg):
                indices.append(i)
                matched = True
                break
        
        if not matched:
            # Create new group
            unique_groups.append((config, [i]))
    
    return unique_groups

if analyze_btn:
    st.header("📊 Configuration Analysis (100 Simulations)")
    
    # Per-hinge weights (uniform for now)
    weights = jnp.ones(hinges.shape[0], dtype=jnp.float32)
    
    # Build total energy function
    hinge_energy_fn = make_hinge_energy_fn(faces, hinges, hinge_state, weights, learned_energy_fn)
    hinge_grad = jax.grad(hinge_energy_fn)
    
    # Run 100 simulations
    st.info("Running 100 simulations with seeds 1-100...")
    progress_bar = st.progress(0)
    
    all_final_states = []
    all_final_angles = []
    
    for seed_idx in range(1, 101):
        # Update progress
        progress_bar.progress(seed_idx / 100)
        
        # Run simulation
        key = jax.random.PRNGKey(seed_idx)
        traj = simulate(
            key, vertices, faces, edges, hinges, hinge_state,
            steps=int(steps), dt=float(dt), sigma=float(sigma),
            theta_gain=float(theta_gain), w_col=float(w_col),
            proj_steps=int(proj_steps), proj_lr=float(proj_lr),
            hinge_grad=hinge_grad,
            method=integration_method
        )
        
        # Get final state
        X_final = np.asarray(traj[-1])
        
        # Compute angles (use robust method to get consistent ordering)
        angles_0_2pi, faces_oriented, hinges_ordered = compute_dihedrals_robust(X_final, faces)
        angles_deg = normalize_angles_to_degrees(angles_0_2pi)
        
        all_final_states.append(X_final)
        all_final_angles.append(angles_deg)
    
    progress_bar.empty()
    st.success("✅ Completed 100 simulations!")
    
    # Group configurations
    st.info("Grouping unique configurations (removing rotational duplicates)...")
    unique_groups = group_configurations(all_final_angles, tolerance_deg=float(angle_tolerance))
    
    # Sort by frequency (most common first)
    unique_groups.sort(key=lambda x: len(x[1]), reverse=True)
    
    st.success(f"✅ Found {len(unique_groups)} unique configurations!")
    
    # Display results
    st.subheader(f"Unique Configurations (Total: {len(unique_groups)})")
    
    # Create columns for display
    n_configs = len(unique_groups)
    
    # Display each configuration
    for config_idx, (representative_config, sim_indices) in enumerate(unique_groups):
        st.markdown(f"---")
        st.markdown(f"### Configuration #{config_idx + 1}")
        
        # Create 3 columns: angles, visualization, stats
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("**📐 Hinge Angles (degrees)**")
            # Display angles as a formatted list
            angles_str = ", ".join([f"{a:.1f}°" for a in representative_config])
            st.text(angles_str)
            
            st.markdown("**🔢 Statistics**")
            count = len(sim_indices)
            percentage = (count / 100.0) * 100
            st.metric("Occurrences", f"{count} / 100")
            st.metric("Percentage", f"{percentage:.1f}%")
            
            # Show which seeds produced this config
            with st.expander("Seeds"):
                seeds_str = ", ".join([str(i+1) for i in sim_indices[:20]])
                if len(sim_indices) > 20:
                    seeds_str += f"... (+{len(sim_indices)-20} more)"
                st.text(seeds_str)
        
        with col2:
            st.markdown("**🎨 3D Visualization**")
            # Get one representative final state
            repr_state_idx = sim_indices[0]
            X_repr = all_final_states[repr_state_idx]
            
            # Use robust method to get properly oriented faces and hinges
            angles_repr, faces_oriented_repr, hinges_ordered_repr = compute_dihedrals_robust(X_repr, faces)
            
            # Create visualization
            config_fig = plot_hinge_angles(X_repr, faces_oriented_repr, hinges_ordered_repr, angles_repr)
            config_fig.update_layout(height=400, width=600)
            st.plotly_chart(config_fig, use_container_width=True)
        
        with col3:
            st.markdown("**📊 Angle Distribution**")
            # Show histogram of angles for this config
            import plotly.graph_objects as go
            angle_hist = go.Figure(data=[go.Histogram(
                x=representative_config,
                nbinsx=12,
                marker_color='lightblue'
            )])
            angle_hist.update_layout(
                xaxis_title="Angle (deg)",
                yaxis_title="Count",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(angle_hist, use_container_width=True)
