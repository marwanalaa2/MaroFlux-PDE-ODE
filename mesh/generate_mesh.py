# mesh/generate_mesh.py
"""
Generate a 3D mesh for the MaroFlux PDE simulations using Gmsh and FEniCS.

The domain is a cube (tissue) containing a cylindrical scaffold with a
central macropore. The mesh is refined near the scaffold-tissue interface.

Usage:
    python generate_mesh.py --output mesh_files/

Author: Marwan Alaa
"""

import argparse
import os
import sys

try:
    import gmsh
except ImportError:
    print("Error: gmsh is required. Install with: pip install gmsh")
    sys.exit(1)

try:
    from dolfin import Mesh, MeshFunction, File, HDF5File, MPI
except ImportError:
    print("Warning: FEniCS not found. Only Gmsh .msh file will be generated.")


def create_mesh(
    L=5.0,
    scaffold_radius=1.0,
    macropore_radius=0.2,
    mesh_size_factor=1.0,
    output_dir="mesh_files",
):
    """
    Create a 3D mesh of a cubic tissue domain with an embedded cylindrical scaffold.

    Parameters
    ----------
    L : float
        Side length of the cubic domain (mm).
    scaffold_radius : float
        Outer radius of the cylindrical scaffold (mm).
    macropore_radius : float
        Radius of the central macropore (mm).
    mesh_size_factor : float
        Global mesh size multiplier (1.0 = default, <1 = finer).
    output_dir : str
        Directory to save mesh files.
    """
    os.makedirs(output_dir, exist_ok=True)

    gmsh.initialize()
    gmsh.model.add("maroflux_domain")

    # Characteristic lengths
    lc_tissue = 0.1 * mesh_size_factor       # 100 µm in bulk tissue
    lc_interface = 0.02 * mesh_size_factor    # 20 µm at scaffold interface
    lc_scaffold = 0.05 * mesh_size_factor     # 50 µm in scaffold strut

    # -------------------------------------------------------------------------
    # 1. Define geometry
    # -------------------------------------------------------------------------
    # Outer tissue box (corner at origin, extends to L)
    box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)

    # Scaffold cylinder (centered in box, spanning full height)
    cx, cy = L / 2, L / 2
    scaffold = gmsh.model.occ.addCylinder(cx, cy, 0, 0, 0, L, scaffold_radius)

    # Macropore cylinder (removed from scaffold)
    macropore = gmsh.model.occ.addCylinder(cx, cy, 0, 0, 0, L, macropore_radius)

    # Boolean operations:
    # tissue = box - scaffold
    # scaffold_strut = scaffold - macropore
    tissue_tag = gmsh.model.occ.cut([(3, box)], [(3, scaffold)])[0][0][1]
    scaffold_tag = gmsh.model.occ.cut([(3, scaffold)], [(3, macropore)])[0][0][1]

    gmsh.model.occ.synchronize()

    # -------------------------------------------------------------------------
    # 2. Physical groups (for boundary conditions)
    # -------------------------------------------------------------------------
    # Volume markers: tissue = 1, scaffold = 2
    gmsh.model.addPhysicalGroup(3, [tissue_tag], 1)
    gmsh.model.setPhysicalName(3, 1, "tissue")

    gmsh.model.addPhysicalGroup(3, [scaffold_tag], 2)
    gmsh.model.setPhysicalName(3, 2, "scaffold")

    # Outer boundary surfaces (for Dirichlet / zero-flux conditions)
    outer_surfaces = []
    all_surfaces = gmsh.model.getBoundary([(3, tissue_tag)], recursive=False)
    for surf in all_surfaces:
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        # Outer faces are at x=0, x=L, y=0, y=L, z=0, z=L
        if any(abs(c - 0) < 1e-6 or abs(c - L) < 1e-6 for c in com):
            outer_surfaces.append(surf[1])

    gmsh.model.addPhysicalGroup(2, outer_surfaces, 10)
    gmsh.model.setPhysicalName(2, 10, "outer_boundary")

    # Scaffold-tissue interface
    scaffold_surfaces = gmsh.model.getBoundary([(3, scaffold_tag)], recursive=False)
    interface_surfaces = [s[1] for s in scaffold_surfaces]
    gmsh.model.addPhysicalGroup(2, interface_surfaces, 20)
    gmsh.model.setPhysicalName(2, 20, "scaffold_interface")

    # -------------------------------------------------------------------------
    # 3. Mesh size fields
    # -------------------------------------------------------------------------
    # Distance field from scaffold interface (for refinement)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", interface_surfaces)

    # Threshold: refined near interface, coarse far away
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_interface)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_tissue)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0)

    # Set as background mesh size
    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    # -------------------------------------------------------------------------
    # 4. Generate mesh
    # -------------------------------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)

    # -------------------------------------------------------------------------
    # 5. Save
    # -------------------------------------------------------------------------
    msh_path = os.path.join(output_dir, "maroflux_mesh.msh")
    gmsh.write(msh_path)
    print(f"Gmsh mesh saved to {msh_path}")

    # -------------------------------------------------------------------------
    # 6. Convert to FEniCS XDMF (if dolfin is available)
    # -------------------------------------------------------------------------
    try:
        mesh = Mesh()
        with XDMFFile(msh_path) as f:  # Actually, FEniCS reads .msh via meshio or dolfin-convert
            pass  # Legacy approach; modern FEniCS uses meshio

        # Alternative: use meshio to convert .msh to .xdmf
        import meshio

        msh_data = meshio.read(msh_path)

        # Extract tetrahedral cells
        tetra_cells = None
        for cell_block in msh_data.cells:
            if cell_block.type == "tetra":
                tetra_cells = cell_block.data
                break

        if tetra_cells is None:
            raise ValueError("No tetrahedral cells found in mesh.")

        # Write XDMF for FEniCS
        xdmf_path = os.path.join(output_dir, "maroflux_mesh.xdmf")
        meshio.write(xdmf_path, meshio.Mesh(
            points=msh_data.points,
            cells=[("tetra", tetra_cells)],
            cell_data={"f": [msh_data.cell_data["gmsh:physical"][0]]},
        ))
        print(f"FEniCS XDMF mesh saved to {xdmf_path}")

    except ImportError:
        print("meshio not found. Skipping XDMF conversion.")

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MaroFlux 3D mesh.")
    parser.add_argument("--L", type=float, default=5.0, help="Domain side length (mm)")
    parser.add_argument("--scaffold-radius", type=float, default=1.0, help="Scaffold radius (mm)")
    parser.add_argument("--macropore-radius", type=float, default=0.2, help="Macropore radius (mm)")
    parser.add_argument("--mesh-size-factor", type=float, default=1.0, help="Mesh size factor")
    parser.add_argument("--output", default="mesh_files", help="Output directory")
    args = parser.parse_args()

    create_mesh(
        L=args.L,
        scaffold_radius=args.scaffold_radius,
        macropore_radius=args.macropore_radius,
        mesh_size_factor=args.mesh_size_factor,
        output_dir=args.output,
    )
