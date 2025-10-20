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
n_sims = st.sidebar.number_input("Number of simulations", 10, 500, 100, 10, help="Number of simulations to run for configuration analysis")
analyze_btn = st.sidebar.button("Run Configuration Analysis", help="Run multiple simulations with different seeds to identify unique fold configurations (accounting for symmetries)")

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

def classify_fold_direction(angle_deg, flat_angle=180.0, threshold_percent=0.1):
    """
    Classify fold direction based on angle.
    
    Args:
        angle_deg: Angle in degrees (0-360)
        flat_angle: Reference flat angle (default 180°)
        threshold_percent: Percentage threshold for flat (default 10%)
    
    Returns:
        +1 for outward fold, 0 for flat, -1 for inward fold
    """
    threshold = flat_angle * threshold_percent
    
    if angle_deg > flat_angle + threshold:
        return 1  # Outward
    elif angle_deg < flat_angle - threshold:
        return -1  # Inward
    else:
        return 0  # Flat

def normalize_angles_to_degrees(angles):
    """Convert angles to degrees in [0, 360) range."""
    return np.mod(np.degrees(angles), 360.0)

def angles_to_state_vector(angles_deg):
    """Convert angle array to discrete state vector [-1, 0, +1]."""
    return np.array([classify_fold_direction(a) for a in angles_deg], dtype=int)

def get_canonical_form(state_vector):
    """
    Get canonical form accounting for cyclic permutation and sign-flip symmetries.
    
    Returns the lexicographically smallest representation among all:
    - Cyclic rotations
    - Sign-flipped versions of all rotations
    """
    n = len(state_vector)
    candidates = []
    
    # Generate all cyclic rotations
    for shift in range(n):
        rotated = np.roll(state_vector, shift)
        candidates.append(tuple(rotated))
        # Also add sign-flipped version
        flipped = -rotated
        candidates.append(tuple(flipped))
    
    # Return lexicographically smallest (canonical form)
    return min(candidates)

def group_configurations(all_state_vectors):
    """
    Group configurations by canonical form, accounting for cyclic and sign-flip symmetries.
    
    Args:
        all_state_vectors: List of state vectors (each is array of -1/0/+1)
    
    Returns:
        List of tuples: (canonical_form, list_of_indices)
    """
    canonical_to_indices = {}
    
    for i, state_vec in enumerate(all_state_vectors):
        canonical = get_canonical_form(state_vec)
        
        if canonical not in canonical_to_indices:
            canonical_to_indices[canonical] = []
        canonical_to_indices[canonical].append(i)
    
    # Convert to list of tuples
    return [(np.array(canonical), indices) for canonical, indices in canonical_to_indices.items()]

if analyze_btn:
    n_sims_int = int(n_sims)
    st.header(f"📊 Configuration Analysis ({n_sims_int} Simulations)")
    
    # Per-hinge weights (uniform for now)
    weights = jnp.ones(hinges.shape[0], dtype=jnp.float32)
    
    # Build total energy function
    hinge_energy_fn = make_hinge_energy_fn(faces, hinges, hinge_state, weights, learned_energy_fn)
    hinge_grad = jax.grad(hinge_energy_fn)
    
    # Run simulations
    st.info(f"Running {n_sims_int} simulations with seeds 1-{n_sims_int}...")
    progress_bar = st.progress(0)
    
    all_final_states = []
    all_state_vectors = []
    
    for seed_idx in range(1, n_sims_int + 1):
        # Update progress
        progress_bar.progress(seed_idx / n_sims_int)
        
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
        
        # Convert to discrete state vector
        state_vec = angles_to_state_vector(angles_deg)
        
        all_final_states.append(X_final)
        all_state_vectors.append(state_vec)
    
    progress_bar.empty()
    st.success(f"✅ Completed {n_sims_int} simulations!")
    
    # Group configurations by canonical form (with symmetries)
    st.info("Grouping unique configurations (accounting for cyclic and sign-flip symmetries)...")
    unique_groups = group_configurations(all_state_vectors)
    
    # Sort by frequency (most common first)
    unique_groups.sort(key=lambda x: len(x[1]), reverse=True)
    
    st.success(f"✅ Found {len(unique_groups)} unique configurations!")
    
    # Display results
    st.subheader(f"Unique Configurations (Total: {len(unique_groups)})")
    st.caption("Fold states: +1=Outward, 0=Flat (±10%), -1=Inward")
    
    # Display each configuration
    for config_idx, (canonical_state, sim_indices) in enumerate(unique_groups):
        st.markdown(f"---")
        st.markdown(f"### Configuration #{config_idx + 1}")
        
        # Create 2 columns: state/stats and visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📐 Fold State Pattern**")
            # Display state vector with symbols
            state_symbols = {-1: "⬇ Inward", 0: "➖ Flat", 1: "⬆ Outward"}
            state_str = ", ".join([f"{s:+d}" for s in canonical_state])
            st.code(state_str)
            
            # Show symbolic representation
            with st.expander("Details"):
                for i, s in enumerate(canonical_state):
                    st.text(f"Hinge {i+1}: {state_symbols[s]}")
            
            st.markdown("**🔢 Statistics**")
            count = len(sim_indices)
            st.metric("Occurrences", f"{count} / {n_sims_int}")
            
            # Show which seeds produced this config
            with st.expander("Seeds"):
                seeds_str = ", ".join([str(i+1) for i in sim_indices[:20]])
                if len(sim_indices) > 20:
                    seeds_str += f"... (+{len(sim_indices)-20} more)"
                st.text(seeds_str)
        
        with col2:
            st.markdown("**🎨 3D Visualization (Representative)**")
            # Get one representative final state
            repr_state_idx = sim_indices[0]
            X_repr = all_final_states[repr_state_idx]
            
            # Use robust method to get properly oriented faces and hinges
            angles_repr, faces_oriented_repr, hinges_ordered_repr = compute_dihedrals_robust(X_repr, faces)
            
            # Create visualization
            config_fig = plot_hinge_angles(X_repr, faces_oriented_repr, hinges_ordered_repr, angles_repr)
            config_fig.update_layout(height=400, width=600)
            st.plotly_chart(config_fig, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Configuration Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .config {{ border: 1px solid #ddd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .state {{ background: #f0f0f0; padding: 10px; font-family: monospace; }}
            .stats {{ margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Configuration Analysis Report</h1>
        <p><strong>Total Simulations:</strong> {n_sims_int}</p>
        <p><strong>Unique Configurations:</strong> {len(unique_groups)}</p>
        <p><strong>Fold State Classification:</strong> +1=Outward, 0=Flat (±10%), -1=Inward</p>
        
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Config #</th>
                <th>Fold State Pattern</th>
                <th>Occurrences</th>
            </tr>
    """
    
    for config_idx, (canonical_state, sim_indices) in enumerate(unique_groups):
        state_str = ", ".join([f"{s:+d}" for s in canonical_state])
        count = len(sim_indices)
        html_report += f"""
            <tr>
                <td>{config_idx + 1}</td>
                <td class="state">{state_str}</td>
                <td>{count} / {n_sims_int}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Detailed Configurations</h2>
    """
    
    for config_idx, (canonical_state, sim_indices) in enumerate(unique_groups):
        state_str = ", ".join([f"{s:+d}" for s in canonical_state])
        count = len(sim_indices)
        state_symbols = {-1: "⬇ Inward", 0: "➖ Flat", 1: "⬆ Outward"}
        
        # Get representative state and create visualization
        repr_state_idx = sim_indices[0]
        X_repr = all_final_states[repr_state_idx]
        angles_repr, faces_oriented_repr, hinges_ordered_repr = compute_dihedrals_robust(X_repr, faces)
        config_fig = plot_hinge_angles(X_repr, faces_oriented_repr, hinges_ordered_repr, angles_repr)
        config_fig.update_layout(height=400, width=600)
        
        # Convert Plotly figure to HTML div
        fig_html = config_fig.to_html(include_plotlyjs='cdn', div_id=f"plot_{config_idx}")
        
        html_report += f"""
        <div class="config">
            <h3>Configuration #{config_idx + 1}</h3>
            <table style="width: 100%; border: none;">
                <tr>
                    <td style="width: 50%; vertical-align: top; border: none;">
                        <div class="state">{state_str}</div>
                        <div class="stats">
                            <strong>Occurrences:</strong> {count} / {n_sims_int}<br>
                            <strong>Seeds:</strong> {", ".join([str(i+1) for i in sim_indices[:20]])}{"..." if len(sim_indices) > 20 else ""}
                        </div>
                        <p><strong>Hinge Details:</strong></p>
                        <ul>
        """
        
        for i, s in enumerate(canonical_state):
            html_report += f"<li>Hinge {i+1}: {state_symbols[s]}</li>"
        
        html_report += f"""
                        </ul>
                    </td>
                    <td style="width: 50%; vertical-align: top; border: none;">
                        <div style="width: 100%; height: 400px;">
                            {fig_html}
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        """
    
    html_report += """
    </body>
    </html>
    """
    
    # Generate text report
    pdf_text_report = f"""Configuration Analysis Report
{'='*50}

Total Simulations: {n_sims_int}
Unique Configurations: {len(unique_groups)}
Fold State Classification: +1=Outward, 0=Flat (±10%), -1=Inward

{'='*50}
SUMMARY
{'='*50}

"""
    
    for config_idx, (canonical_state, sim_indices) in enumerate(unique_groups):
        state_str = ", ".join([f"{s:+d}" for s in canonical_state])
        count = len(sim_indices)
        pdf_text_report += f"""Configuration #{config_idx + 1}
Fold State Pattern: {state_str}
Occurrences: {count} / {n_sims_int}
Seeds: {", ".join([str(i+1) for i in sim_indices[:20]])}{"..." if len(sim_indices) > 20 else ""}

"""
    
    # Download buttons with unique keys to prevent app reset
    col_html, col_pdf = st.columns(2)
    
    with col_html:
        st.download_button(
            label="📄 Download HTML Report",
            data=html_report,
            file_name="configuration_analysis.html",
            mime="text/html",
            key="download_html_report"
        )
    
    with col_pdf:
        st.download_button(
            label="📑 Download Text Report (PDF-compatible)",
            data=pdf_text_report,
            file_name="configuration_analysis.txt",
            mime="text/plain",
            key="download_text_report"
        )
