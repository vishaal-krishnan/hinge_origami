import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re 


# SciPy for spline/integration
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid

# JAX + Optax
import jax
import jax.numpy as jnp
import optax

from pathlib import Path

# Path to bundled default torque file
DEFAULT_TORQUE_PATH = Path(__file__).parent / "data" / "unit075_torque.txt"


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Origami Hinge Simulator", layout="wide")
st.title("Origami Hinge Simulator")

st.sidebar.header("1) Upload torque–angle data")
uploaded = st.sidebar.file_uploader("File (.txt/.csv): two columns (Angle, Torque)", type=["txt", "csv"])

delim = st.sidebar.selectbox("Delimiter", ["auto", "space", "comma", "tab"], index=0)
smoothing = st.sidebar.slider("Spline smoothing (s)", 0.0, 5.0, 0.5, 0.1)


# === Sidebar controls ===
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
    p_sum = p_neg + p_zero + p_pos
    if p_sum == 0:
        st.sidebar.warning("Sum of probabilities is 0; normalizing to equal thirds.")
        p_neg = p_zero = p_pos = 1/3
    else:
        p_neg, p_zero, p_pos = np.array([p_neg, p_zero, p_pos]) / p_sum

    state_seed = st.sidebar.number_input("Seed (hinge-state sampling)", value=0, step=1)  # NEW
else:
    # Manual assignment UI
    st.sidebar.markdown("**Manual assignment**")
    default_all = st.sidebar.selectbox("Set ALL hinges to:", [-1, 0, 1], index=1)
    manual_text = st.sidebar.text_area(
        "Optional: paste custom list (-1 0 1 ...), separated by space/comma/newline",
        value=""
    )
    state_seed = None  # not used in manual mode

st.sidebar.header("3) Simulation")
steps = st.sidebar.number_input("Steps", 100, 10000, 1000, 100)
dt = st.sidebar.number_input("dt", 1e-5, 1.0, 0.01, 0.01, format="%.5f")
sigma = st.sidebar.number_input("Noise σ", 0.0, 0.1, 1e-4, 1e-4, format="%.6f")
theta_gain = st.sidebar.number_input("Theta gain (force scale)", 0.0, 10.0, 1.0, 0.1)
w_col = st.sidebar.number_input("Collision weight", 0.0, 100.0, 10.0, 1.0)
proj_steps = st.sidebar.number_input("Isometry proj steps", 0, 1000, 50, 10)
proj_lr = st.sidebar.number_input("Isometry proj lr", 1e-5, 1e-1, 1e-2, 1e-3, format="%.5f")
radius = st.sidebar.number_input("Hex radius (tiles)", 0, 3, 0, 1)

sim_seed = st.sidebar.number_input("Seed (simulation noise)", value=1, step=1)  # NEW
run_btn = st.sidebar.button("Run Simulation")



# -----------------------
# Utilities: Mesh / Geometry
# -----------------------
def rotation_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                      [jnp.sin(theta),  jnp.cos(theta)]])

def create_triangle():
    L = 1.0
    p0 = jnp.array([0.0, 0.0])
    p1 = jnp.array([L, 0.0])
    p2 = jnp.array([L/2, jnp.sqrt(3)/2 * L])
    return jnp.stack([p0, p1, p2])

def create_hex_tile():
    base = create_triangle()
    return jnp.stack([base @ rotation_matrix(i * jnp.pi/3).T for i in range(6)])  # (6,3,2)

def axial_to_cart(q, r):
    x = 3/2 * q
    y = jnp.sqrt(3) * (r + q/2)
    return jnp.array([x, y])

def create_hex_grid(radius=1):
    coords = []
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                coords.append(axial_to_cart(q, r))
    return jnp.stack(coords) if coords else jnp.zeros((0,2))

def create_mesh(radius=1):
    hex_tile = create_hex_tile()
    centers = create_hex_grid(radius)
    vertices, faces, vmap = [], [], {}
    vid = 0
    for center in centers:
        for tri in hex_tile:
            tri_3d = jnp.column_stack((tri + center, jnp.zeros(3)))
            ids = []
            for v in tri_3d:
                key = tuple(np.round(np.array(v), 4))
                if key not in vmap:
                    vmap[key] = vid
                    vertices.append(np.array(v))
                    vid += 1
                ids.append(vmap[key])
            faces.append(tuple(ids))
    if len(vertices)==0:
        # fallback: single hex at origin
        tri_3d = jnp.column_stack((create_triangle(), jnp.zeros(3)))
        vertices = [np.array(v) for v in np.array(tri_3d)]
        faces = [tuple([0,1,2])]
    return jnp.array(vertices), jnp.array(faces)

def compute_edges(faces):
    edge_set = set()
    for tri in faces:
        a, b, c = map(int, tri)
        edge_set |= {tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))}
    return list(edge_set)

@jax.jit
def face_normals(X, faces):
    v0, v1, v2 = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    n = jnp.cross(v1 - v0, v2 - v0)
    return n / (jnp.linalg.norm(n, axis=1, keepdims=True) + 1e-6)

def build_hinge_graph(faces):
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for i, tri in enumerate(faces):
        a, b, c = map(int, tri)
        for e in [(a,b), (b,c), (c,a)]:
            edge_to_faces[tuple(sorted(e))].append(i)
    return jnp.array([tuple(v) for v in edge_to_faces.values() if len(v) == 2])

# Robust dihedral helpers (coherent orientation pipeline)
def faces_to_edges_oriented(f): return [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
def faces_to_edges_unordered(f):
    return [(min(f[0], f[1]), max(f[0], f[1])),
            (min(f[1], f[2]), max(f[1], f[2])),
            (min(f[2], f[0]), max(f[2], f[0]))]

def build_face_adjacency(faces):
    F = np.asarray(faces, dtype=int)
    edge_map = {}
    for fi, f in enumerate(F):
        e_or = faces_to_edges_oriented(f)
        e_un = faces_to_edges_unordered(f)
        for k in range(3):
            key = e_un[k]
            edge_map.setdefault(key, []).append((fi, k, e_or[k]))
    nbrs = [[] for _ in range(len(F))]
    shared_edge_info = {}
    for key, lst in edge_map.items():
        if len(lst) == 2:
            (f1,k1,uv1), (f2,k2,uv2) = lst
            nbrs[f1].append((f2, k1, k2, uv1, uv2, key))
            nbrs[f2].append((f1, k2, k1, uv2, uv1, key))
            shared_edge_info[key] = ((f1,k1,uv1), (f2,k2,uv2))
    return nbrs, shared_edge_info

def orient_faces_coherently(faces):
    F = np.asarray(faces, dtype=int).copy()
    nbrs, _ = build_face_adjacency(F)
    nF = len(F); visited = np.zeros(nF, dtype=bool); flipped = np.zeros(nF, dtype=bool)
    for root in range(nF):
        if visited[root]: continue
        visited[root] = True; stack = [root]
        while stack:
            fi = stack.pop()
            for (fj, k_i, k_j, uv_i, uv_j, key) in nbrs[fi]:
                if visited[fj]: continue
                need_flip = (uv_j == uv_i)
                if need_flip:
                    F[fj] = F[fj][[0,2,1]]
                    flipped[fj] = ~flipped[fj]
                visited[fj] = True; stack.append(fj)
    return F, flipped

def face_normals_np(X, faces):
    v0 = X[faces[:,0]]; v1 = X[faces[:,1]]; v2 = X[faces[:,2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n

def build_hinges_ordered(faces):
    F = np.asarray(faces, dtype=int)
    _, shared = build_face_adjacency(F)
    hinges = []; edge_uv_in_f1 = []
    for key, ((f1,k1,uv1),(f2,k2,uv2)) in shared.items():
        hinges.append((f1, f2)); edge_uv_in_f1.append(uv1)
    return np.asarray(hinges, dtype=int), np.asarray(edge_uv_in_f1, dtype=int)

def dihedral_one_sided(X, faces, hinges, edge_uv_in_f1):
    X = np.asarray(X); F = np.asarray(faces, dtype=int)
    H = np.asarray(hinges, dtype=int); UV = np.asarray(edge_uv_in_f1, dtype=int)
    N = face_normals_np(X, F); n1 = N[H[:,0]]; n2 = N[H[:,1]]
    u = UV[:,0]; v = UV[:,1]
    e = X[v] - X[u]; e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    s = np.einsum('ij,ij->i', e, np.cross(n1, n2))
    c = np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0)
    phi = np.arctan2(s, c)
    return (np.pi - phi) % (2.0 * np.pi)

def compute_dihedrals_robust(X, faces):
    faces_oriented, _ = orient_faces_coherently(faces)
    hinges_ordered, edge_uv_in_f1 = build_hinges_ordered(faces_oriented)
    angles = dihedral_one_sided(X, faces_oriented, hinges_ordered, edge_uv_in_f1)
    return angles, faces_oriented, hinges_ordered

# -----------------------
# Isometry + dynamics
# -----------------------
@jax.jit
def compute_metric(X, edges):
    return jnp.sum((X[edges[:, 0]] - X[edges[:, 1]])**2, axis=-1)

@jax.jit
def isometry_loss(X, edges, initial_metric):
    return jnp.mean((compute_metric(X, edges) - initial_metric) ** 2)

@jax.jit
def collision_penalty(X, r=0.5):
    D2 = jnp.sum((X[:, None] - X[None])**2, axis=-1)
    mask = (D2 > 1e-6) & (D2 < r**2)
    return jnp.sum(jnp.where(mask, (r**2 - D2)**2, 0.0)) / 2

collision_force = jax.grad(collision_penalty)

def project_isometry(X, V, edges, metric, steps=10, lr=0.05):
    opt = optax.adam(lr)
    state = opt.init(V)
    def loss_fn(V): return jnp.sum((compute_metric(X + V, edges) - metric)**2)
    def step(carry, _):
        V, opt_state = carry
        loss, grad = jax.value_and_grad(loss_fn)(V)
        updates, opt_state = opt.update(grad, opt_state)
        return (optax.apply_updates(V, updates), opt_state), loss
    (V_final, _), _ = jax.lax.scan(step, (V, state), None, length=steps)
    return V_final

# Placeholder for energy lookup (set after upload)
theta_exp = None; E_exp = None

def energy_lookup(theta):
    theta_abs = jnp.abs(theta)
    theta_clamped = jnp.clip(theta_abs, theta_exp[0], theta_exp[-1])
    return jnp.interp(theta_clamped, theta_exp, E_exp)

def hinge_bending_energy_learned(X, faces, hinges, state, theta_0, learned_energy_fn):
    """
    Compute hinge bending energy using a learned energy function instead of quadratic potential.

    Parameters:
        X : (V, 3) vertex positions
        faces : (F, 3) face indices
        hinges : (H, 2) face pairs
        state : (H,) in {-1, 0, +1} specifying desired fold direction
        theta_0 : target fold angle (float)
        learned_energy_fn : callable that maps (angle,) → energy

    Returns:
        total energy (scalar)
    """
    N = face_normals(X, faces)
    n1, n2 = N[hinges[:, 0]], N[hinges[:, 1]]

    # Unsigned angle
    dot = jnp.clip(jnp.sum(n1 * n2, axis=1), -1.0, 1.0)
    theta = jnp.arccos(dot)

    # Sign from Z-component of cross product (2.5D convention)
    cross = jnp.cross(n1, n2)
    sign = jnp.sign(cross[:, 2])
    signed_theta = theta * sign

    # Target angle (same sign as state)
    theta_star = state * theta_0

    # Difference from target
    dtheta = signed_theta - theta_star

    # Determine soft/stiff behavior based on whether fold follows the state
    follows_state = jnp.sign(dtheta) == jnp.sign(state)
    weights = jnp.where(state == 0, 1.0, jnp.where(follows_state, 0.5, 1.0))

    # Apply learned energy
    energy = learned_energy_fn(signed_theta)

    return jnp.sum(weights * energy)

hinge_grad = jax.grad(hinge_bending_energy, argnums=0)

def diffusion_step(X, key, edges, faces, hinges, state, metric, dt, theta_gain, sigma, w_col, proj_steps, proj_lr):
    drift = -theta_gain * hinge_grad(X, faces, hinges, state, 0.0, None, None, None)
    noise = sigma * jax.random.normal(key, X.shape)
    V = drift * dt + noise * jnp.sqrt(dt)
    V_proj = project_isometry(X, V, edges, metric, proj_steps, proj_lr)
    col = w_col * collision_force(X + V_proj) * dt
    return X + V_proj + col

def simulate(key, X0, faces, edges, hinges, hinge_state, steps, dt, sigma, theta_gain, w_col, proj_steps, proj_lr):
    metric0 = compute_metric(X0, edges)
    def step(X, key):
        key, sub = jax.random.split(key)
        X_new = diffusion_step(X, sub, edges, faces, hinges, hinge_state, metric0,
                               dt, theta_gain, sigma, w_col, proj_steps, proj_lr)
        return X_new, X_new
    keys = jax.random.split(key, steps)
    _, traj = jax.lax.scan(step, X0, keys)
    return jnp.concatenate([X0[None], traj], axis=0)

# -----------------------
# Plot helpers
# -----------------------


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_torque_stiffness_energy(angle_smooth, torque_smooth, slope_smooth, energy_smooth):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{}]]
    )

    # Row 1: Torque (left) and Stiffness (right)
    fig.add_trace(go.Scatter(x=angle_smooth, y=torque_smooth, name="Torque"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=angle_smooth, y=slope_smooth,  name="Stiffness (dτ/dθ)"), row=1, col=1, secondary_y=True)

    # Row 2: Energy
    fig.add_trace(go.Scatter(x=angle_smooth, y=energy_smooth, name="Energy"), row=2, col=1)

    # Axes titles
    fig.update_yaxes(title_text="Torque", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Stiffness", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    fig.update_xaxes(title_text="Angle [rad]", row=2, col=1)

    fig.update_layout(height=600, width=900, legend=dict(orientation="h"))
    return fig


def create_plotly_animated_trajectory(trajectory, faces, edges, frame_stride=10):
    trajectory = np.asarray(trajectory); faces = np.asarray(faces); edges = np.asarray(edges)
    T = trajectory.shape[0]; fig = go.Figure()
    mesh_kwargs = dict(opacity=0.5, color='lightblue')

    # Initial
    X0 = trajectory[0]
    fig.add_trace(go.Mesh3d(x=X0[:,0], y=X0[:,1], z=X0[:,2],
                            i=faces[:,0], j=faces[:,1], k=faces[:,2],
                            name='mesh', **mesh_kwargs))
    # edges
    edge_indices = edges
    edge_xyz = X0[edge_indices]
    edge_xyz = np.insert(edge_xyz, 2, np.nan, axis=1).reshape(-1, 3)
    fig.add_trace(go.Scatter3d(x=edge_xyz[:,0], y=edge_xyz[:,1], z=edge_xyz[:,2],
                               mode='lines', line=dict(color='black', width=2),
                               name='edges', showlegend=False))

    frames=[]
    for t in range(0, T, frame_stride):
        Xt = trajectory[t]
        edge_xyz_t = Xt[edge_indices]
        edge_xyz_t = np.insert(edge_xyz_t, 2, np.nan, axis=1).reshape(-1, 3)
        frames.append(go.Frame(data=[
            go.Mesh3d(x=Xt[:,0], y=Xt[:,1], z=Xt[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], name='mesh', **mesh_kwargs),
            go.Scatter3d(x=edge_xyz_t[:,0], y=edge_xyz_t[:,1], z=edge_xyz_t[:,2], mode='lines', line=dict(color='black', width=2), name='edges', showlegend=False)
        ], name=str(t)))
    fig.frames = frames
    fig.update_layout(
        width=800, height=800,
        scene=dict(aspectmode='data'),
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None])])],
        sliders=[{"steps":[{"args":[[f"{t}"],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}],"label":f"{t}","method":"animate"} for t in range(0,T,frame_stride)],
                 "x":0, "y":0, "currentvalue":{"font":{"size":14},"prefix":"Frame: ","visible":True}, "len":1.0}]
    )
    return fig

def plot_hinge_angles(X, faces, hinges, signed_angles):
    X = np.asarray(X); faces = np.asarray(faces); hinges = np.asarray(hinges); A = np.asarray(signed_angles)
    # get shared edge endpoints
    def get_edge(f1, f2):
        s = list(set(faces[f1]) & set(faces[f2]))
        if len(s)!=2: raise ValueError("Expected 2 shared vertices.")
        return X[s[0]], X[s[1]]

    starts, ends = zip(*[get_edge(f1, f2) for f1, f2 in hinges])
    starts = np.array(starts); ends = np.array(ends)
    # colormap over 0..2π
    cmap = plt.get_cmap("coolwarm")
    colors = (np.array([cmap((a%(2*np.pi))/(2*np.pi))[:3] for a in A])*255).astype(int)

    lines=[]
    for i in range(len(starts)):
        lines.append(go.Scatter3d(x=[starts[i][0], ends[i][0]],
                                  y=[starts[i][1], ends[i][1]],
                                  z=[starts[i][2], ends[i][2]],
                                  mode='lines',
                                  line=dict(color=f"rgb({colors[i,0]},{colors[i,1]},{colors[i,2]})", width=6),
                                  hovertext=f"{np.degrees(A[i]):.2f}°", hoverinfo='text', showlegend=False))
    mesh = go.Mesh3d(x=X[:,0], y=X[:,1], z=X[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],
                     opacity=0.5, color='lightgray', name='mesh')
    fig = go.Figure(data=[mesh]+lines)
    fig.update_layout(width=800, height=800, scene=dict(aspectmode='data'),
                      title="Hinge Angles (Color-Coded, 0–360°)")
    return fig

# -----------------------
# Data + Run
# -----------------------
def read_dataframe(uploaded, delim):
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

# --- File -> DataFrame (uploaded OR default) ---
if uploaded:
    df = read_dataframe(uploaded, delim)
    st.success("File uploaded successfully!")
else:
    st.info("Using default torque.txt")
    if DEFAULT_TORQUE_PATH.exists():
        # Try to honor the delimiter choice; default is whitespace
        if delim in ("auto", "space"):
            df = pd.read_csv(DEFAULT_TORQUE_PATH, header=None, sep=r"\s+")
        elif delim == "comma":
            df = pd.read_csv(DEFAULT_TORQUE_PATH, header=None)
        else:  # tab
            df = pd.read_csv(DEFAULT_TORQUE_PATH, header=None, sep="\t")
        df.columns = ["Angle", "Torque"]
    else:
        st.error(f"Default file not found at: {DEFAULT_TORQUE_PATH}")
        st.stop()

# Preview
st.write("Preview:", df.head())

# --- Calibrate spline and energy ---
sort_idx = np.argsort(df['Angle'].values)
angles = df['Angle'].values[sort_idx] * np.pi / 180.0
torques = df['Torque'].values[sort_idx]
spline = UnivariateSpline(angles, torques, s=float(smoothing))
angle_smooth = np.linspace(angles.min(), angles.max(), 500)
torque_smooth = spline(angle_smooth)
slope_smooth = spline.derivative()(angle_smooth)
energy_smooth = cumulative_trapezoid(torque_smooth, angle_smooth, initial=0.0)
energy_smooth -= energy_smooth.min()

# expose to JAX lookup
theta_exp = jnp.array(angle_smooth)
E_exp     = jnp.array(energy_smooth)

st.subheader("Calibration Plots")
st.plotly_chart(
    plot_torque_stiffness_energy(angle_smooth, torque_smooth, slope_smooth, energy_smooth),
    use_container_width=True
)

# --- Build mesh, edges, hinges ---
vertices, faces = create_mesh(radius=int(radius))
edges = jnp.array(compute_edges(faces))
hinges = build_hinge_graph(faces)


# === Hinge state construction (probabilities OR manual) ===
H = int(hinges.shape[0])  # number of hinges
state_vals = jnp.array([-1, 0, 1])

if hinge_mode == "Sample by probabilities":
    probs = jnp.array([p_neg, p_zero, p_pos], dtype=jnp.float32)
    key_states = jax.random.PRNGKey(int(state_seed))  # NEW seed
    hinge_state = jax.random.choice(key_states, state_vals, (H,), p=probs)

else:
    # Manual assignment
    if manual_text.strip():
        # Parse list of ints from text (split on comma/space/newline)
        tokens = [t for t in re.split(r"[,\s]+", manual_text.strip()) if t]
        try:
            vals = np.array([int(t) for t in tokens], dtype=int)
        except ValueError:
            st.error("Manual list contains non-integer entries. Use only -1, 0, or 1.")
            st.stop()
        if np.any(~np.isin(vals, [-1, 0, 1])):
            st.error("Manual list may only contain -1, 0, or 1.")
            st.stop()
        if len(vals) != H:
            st.error(f"Manual list length ({len(vals)}) must match number of hinges ({H}).")
            st.stop()
        hinge_state = jnp.array(vals, dtype=jnp.int32)
    else:
        # Apply one value to all hinges
        hinge_state = jnp.full((H,), int(default_all), dtype=jnp.int32)

# Show a tiny preview
st.caption(f"Hinges: {H} — mode: {hinge_mode}")


# Run simulation
if run_btn:
    st.subheader("Simulation Running…")
    key = jax.random.PRNGKey(int(sim_seed))  # NEW: seed from sidebar
    traj = simulate(key, vertices, faces, edges, hinges, hinge_state,
                steps=int(steps), dt=float(dt), sigma=float(sigma),
                theta_gain=float(theta_gain), w_col=float(w_col),
                proj_steps=int(proj_steps), proj_lr=float(proj_lr))

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
