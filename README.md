# Origami Hinge Simulator

A modular physics simulation for origami hinge dynamics with a Streamlit web interface.

## 📁 Project Structure

```
hinge_origami/
├── app.py                     # Main Streamlit application
├── geometry.py                # Hexagonal mesh + dihedral angles + all geometry ⚙️
├── dynamics.py                # Physics: hinge energy + constraints (isometry + collision) ⭐
├── visualization.py           # 3D plotting and animations
├── data_processing.py         # Load/calibrate torque-angle data
├── hinge_states.py            # UI helpers for hinge state assignment
├── test_dynamics.py           # Simple dynamics testing
├── requirements.txt           # Python dependencies
├── data/unit075_torque.txt    # Default torque-angle data
└── README.md                  # This file
```

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## 🔬 Working on Dynamics Accuracy

The physics code is in `dynamics.py`. To work on it:

1. **Test current physics**:
```bash
python test_dynamics.py
```

2. **Modify `dynamics.py`** - Focus on these functions:
   - `make_hinge_energy_fn()` - Energy function logic
   - `diffusion_step()` - Time integration 
   - `project_isometry()` - Constraint projection

3. **Test changes**:
```bash
python test_dynamics.py
```

4. **Test full app**:
```bash
streamlit run app.py
```

## 📦 Core Modules

### `geometry.py` ⚙️
**Everything about the hexagonal mesh:**
- Hexagonal grid creation (`create_mesh`, `create_hex_grid`)
- Face/edge/vertex operations (`compute_edges`, `build_hinge_graph`)
- Face normals computation (`face_normals`)
- Dihedral angle calculation (`signed_dihedral_angles`, `compute_dihedrals_robust`)
- Face orientation and adjacency (`orient_faces_coherently`)

### `dynamics.py` ⭐ 
**All physics simulation:**
- **Hinge bending energy**: `make_hinge_energy_fn()`, `make_learned_energy_fn()`
- **Constraints**: `project_isometry()` (isometry), `collision_penalty()` (collision avoidance)
- **Time integration**: `diffusion_step()`, `simulate()`
- **Metrics**: `compute_metric()`, `isometry_loss()`

### Supporting Modules
- **`visualization.py`** - 3D animations and calibration plots
- **`data_processing.py`** - Load/calibrate experimental torque-angle data
- **`hinge_states.py`** - UI helpers for hinge state assignment
- **`test_dynamics.py`** - Quick physics testing

## 🎯 Key Features

- Interactive web interface for simulation parameters
- Data-driven energy functions from experimental torque measurements
- 3D animated folding visualization
- **Logical code organization**: geometry vs. physics

The structure now makes sense: `geometry.py` handles everything about the hexagonal mesh and dihedral angles, while `dynamics.py` contains all the physics including hinge energy and constraints.
