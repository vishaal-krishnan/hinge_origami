import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def plot_torque_stiffness_energy(angle_smooth, torque_smooth, slope_smooth, energy_smooth):
    """Plot torque, stiffness, and energy curves."""
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
    """Create animated 3D trajectory visualization."""
    trajectory = np.asarray(trajectory)
    faces = np.asarray(faces)
    edges = np.asarray(edges)
    T = trajectory.shape[0]
    fig = go.Figure()
    mesh_kwargs = dict(opacity=0.5, color='lightblue')

    # Initial frame
    X0 = trajectory[0]
    fig.add_trace(go.Mesh3d(x=X0[:,0], y=X0[:,1], z=X0[:,2],
                            i=faces[:,0], j=faces[:,1], k=faces[:,2],
                            name='mesh', **mesh_kwargs))
    
    # Initial edges
    edge_indices = edges
    edge_xyz = X0[edge_indices]
    edge_xyz = np.insert(edge_xyz, 2, np.nan, axis=1).reshape(-1, 3)
    fig.add_trace(go.Scatter3d(x=edge_xyz[:,0], y=edge_xyz[:,1], z=edge_xyz[:,2],
                               mode='lines', line=dict(color='black', width=2),
                               name='edges', showlegend=False))

    # Animation frames
    frames = []
    for t in range(0, T, frame_stride):
        Xt = trajectory[t]
        edge_xyz_t = Xt[edge_indices]
        edge_xyz_t = np.insert(edge_xyz_t, 2, np.nan, axis=1).reshape(-1, 3)
        frames.append(go.Frame(data=[
            go.Mesh3d(x=Xt[:,0], y=Xt[:,1], z=Xt[:,2], 
                     i=faces[:,0], j=faces[:,1], k=faces[:,2], 
                     name='mesh', **mesh_kwargs),
            go.Scatter3d(x=edge_xyz_t[:,0], y=edge_xyz_t[:,1], z=edge_xyz_t[:,2], 
                        mode='lines', line=dict(color='black', width=2), 
                        name='edges', showlegend=False)
        ], name=str(t)))
    
    fig.frames = frames
    fig.update_layout(
        width=800, height=800,
        scene=dict(aspectmode='data'),
        updatemenus=[dict(type="buttons", showactive=False, 
                         buttons=[dict(label="Play", method="animate", args=[None])])],
        sliders=[{"steps":[{"args":[[f"{t}"],{"frame":{"duration":0,"redraw":True},"mode":"immediate"}],
                          "label":f"{t}","method":"animate"} for t in range(0,T,frame_stride)],
                 "x":0, "y":0, "currentvalue":{"font":{"size":14},"prefix":"Frame: ","visible":True}, "len":1.0}]
    )
    return fig


def plot_hinge_angles(X, faces, hinges, signed_angles):
    """Plot hinge angles with color coding."""
    X = np.asarray(X)
    faces = np.asarray(faces)
    hinges = np.asarray(hinges)
    A = np.asarray(signed_angles)
    
    # Get shared edge endpoints
    def get_edge(f1, f2):
        s = list(set(faces[f1]) & set(faces[f2]))
        if len(s) != 2: 
            raise ValueError("Expected 2 shared vertices.")
        return X[s[0]], X[s[1]]

    starts, ends = zip(*[get_edge(f1, f2) for f1, f2 in hinges])
    starts = np.array(starts)
    ends = np.array(ends)
    
    # Colormap over 0..2π
    cmap = plt.get_cmap("coolwarm")
    colors = (np.array([cmap((a%(2*np.pi))/(2*np.pi))[:3] for a in A])*255).astype(int)

    lines = []
    for i in range(len(starts)):
        lines.append(go.Scatter3d(x=[starts[i][0], ends[i][0]],
                                  y=[starts[i][1], ends[i][1]],
                                  z=[starts[i][2], ends[i][2]],
                                  mode='lines',
                                  line=dict(color=f"rgb({colors[i,0]},{colors[i,1]},{colors[i,2]})", width=6),
                                  hovertext=f"{np.degrees(A[i]):.2f}°", 
                                  hoverinfo='text', showlegend=False))
    
    mesh = go.Mesh3d(x=X[:,0], y=X[:,1], z=X[:,2], 
                     i=faces[:,0], j=faces[:,1], k=faces[:,2],
                     opacity=0.5, color='lightgray', name='mesh')
    
    fig = go.Figure(data=[mesh]+lines)
    fig.update_layout(width=800, height=800, scene=dict(aspectmode='data'),
                      title="Hinge Angles (Color-Coded, 0–360°)")
    return fig 