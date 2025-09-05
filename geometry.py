import numpy as np
import jax.numpy as jnp
import jax
from collections import defaultdict


# ==============================================
# Hexagonal mesh creation
# ==============================================

def rotation_matrix(theta):
    """Create 2D rotation matrix."""
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                      [jnp.sin(theta),  jnp.cos(theta)]])


def create_triangle():
    """Create a basic equilateral triangle."""
    L = 1.0
    p0 = jnp.array([0.0, 0.0])
    p1 = jnp.array([L, 0.0])
    p2 = jnp.array([L/2, jnp.sqrt(3)/2 * L])
    return jnp.stack([p0, p1, p2])


def create_hex_tile():
    """Create hexagonal tile from 6 triangles."""
    base = create_triangle()
    return jnp.stack([base @ rotation_matrix(i * jnp.pi/3).T for i in range(6)])  # (6,3,2)


def axial_to_cart(q, r):
    """Convert axial coordinates to cartesian for hexagonal grid."""
    x = 3/2 * q
    y = jnp.sqrt(3) * (r + q/2)
    return jnp.array([x, y])


def create_hex_grid(radius=1):
    """Create hexagonal grid of centers."""
    coords = []
    for q in range(-radius, radius+1):
        for r in range(-radius, radius+1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= radius:
                coords.append(axial_to_cart(q, r))
    return jnp.stack(coords) if coords else jnp.zeros((0,2))


def create_mesh(radius=1):
    """Create mesh from hexagonal tiling."""
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
    
    if len(vertices) == 0:
        # fallback: single hex at origin
        tri_3d = jnp.column_stack((create_triangle(), jnp.zeros(3)))
        vertices = [np.array(v) for v in np.array(tri_3d)]
        faces = [tuple([0,1,2])]
    
    return jnp.array(vertices), jnp.array(faces)


def compute_edges(faces):
    """Compute unique edges from faces."""
    edge_set = set()
    for tri in faces:
        a, b, c = map(int, tri)
        edge_set |= {tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))}
    return list(edge_set)


# ==============================================
# Face normals and hinges
# ==============================================

@jax.jit
def face_normals(X, faces):
    """Compute face normals (JAX version)."""
    v0, v1, v2 = X[faces[:, 0]], X[faces[:, 1]], X[faces[:, 2]]
    n = jnp.cross(v1 - v0, v2 - v0)
    return n / (jnp.linalg.norm(n, axis=1, keepdims=True) + 1e-6)


def face_normals_np(X, faces):
    """Compute face normals (NumPy version)."""
    v0 = X[faces[:,0]]
    v1 = X[faces[:,1]] 
    v2 = X[faces[:,2]]
    n = np.cross(v1 - v0, v2 - v0)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n


def build_hinge_graph(faces):
    """Build hinge graph from faces (edges shared by exactly 2 faces)."""
    edge_to_faces = defaultdict(list)
    for i, tri in enumerate(faces):
        a, b, c = map(int, tri)
        for e in [(a,b), (b,c), (c,a)]:
            edge_to_faces[tuple(sorted(e))].append(i)
    return jnp.array([tuple(v) for v in edge_to_faces.values() if len(v) == 2])


# ==============================================
# Dihedral angle computation
# ==============================================

def faces_to_edges_oriented(f):
    """Get oriented edges from face."""
    return [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]


def faces_to_edges_unordered(f):
    """Get unordered edges from face."""
    return [(min(f[0], f[1]), max(f[0], f[1])),
            (min(f[1], f[2]), max(f[1], f[2])),
            (min(f[2], f[0]), max(f[2], f[0]))]


def build_face_adjacency(faces):
    """Build face adjacency information."""
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
    """Orient faces coherently to ensure consistent normal directions."""
    F = np.asarray(faces, dtype=int).copy()
    nbrs, _ = build_face_adjacency(F)
    nF = len(F)
    visited = np.zeros(nF, dtype=bool)
    flipped = np.zeros(nF, dtype=bool)
    
    for root in range(nF):
        if visited[root]: 
            continue
        visited[root] = True
        stack = [root]
        
        while stack:
            fi = stack.pop()
            for (fj, k_i, k_j, uv_i, uv_j, key) in nbrs[fi]:
                if visited[fj]: 
                    continue
                need_flip = (uv_j == uv_i)
                if need_flip:
                    F[fj] = F[fj][[0,2,1]]
                    flipped[fj] = ~flipped[fj]
                visited[fj] = True
                stack.append(fj)
    
    return F, flipped


def build_hinges_ordered(faces):
    """Build ordered hinges with edge information."""
    F = np.asarray(faces, dtype=int)
    _, shared = build_face_adjacency(F)
    hinges = []
    edge_uv_in_f1 = []
    
    for key, ((f1,k1,uv1),(f2,k2,uv2)) in shared.items():
        hinges.append((f1, f2))
        edge_uv_in_f1.append(uv1)
    
    return np.asarray(hinges, dtype=int), np.asarray(edge_uv_in_f1, dtype=int)


def dihedral_one_sided(X, faces, hinges, edge_uv_in_f1):
    """Compute one-sided dihedral angles."""
    X = np.asarray(X)
    F = np.asarray(faces, dtype=int)
    H = np.asarray(hinges, dtype=int)
    UV = np.asarray(edge_uv_in_f1, dtype=int)
    
    N = face_normals_np(X, F)
    n1 = N[H[:,0]]
    n2 = N[H[:,1]]
    
    u = UV[:,0]
    v = UV[:,1]
    e = X[v] - X[u]
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    
    s = np.einsum('ij,ij->i', e, np.cross(n1, n2))
    c = np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0)
    phi = np.arctan2(s, c)
    
    return (np.pi - phi) % (2.0 * np.pi)


def compute_dihedrals_robust(X, faces):
    """Compute dihedral angles with robust face orientation."""
    faces_oriented, _ = orient_faces_coherently(faces)
    hinges_ordered, edge_uv_in_f1 = build_hinges_ordered(faces_oriented)
    angles = dihedral_one_sided(X, faces_oriented, hinges_ordered, edge_uv_in_f1)
    return angles, faces_oriented, hinges_ordered


@jax.jit
def signed_dihedral_angles(X, faces, hinges):
    """Compute signed dihedral angles using Z-cross convention (2.5D sheet)."""
    N = face_normals(X, faces)             # (F,3)
    n1 = N[hinges[:, 0]]                   # (H,3)
    n2 = N[hinges[:, 1]]                   # (H,3)
    dot = jnp.clip(jnp.sum(n1 * n2, axis=1), -1.0, 1.0)
    theta = jnp.arccos(dot)                # (H,)
    cross = jnp.cross(n1, n2)              # (H,3)
    sign = jnp.sign(cross[:, 2])           # (H,)
    return theta * sign                    # (H,) 