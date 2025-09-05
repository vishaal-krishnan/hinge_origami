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
