"""Simply-supported Euler-Bernoulli beam model for Z24 bridge validation.

Differences from BeamModel (cantilever):
- Boundary conditions: pinned at both ends (displacement=0, rotation FREE)
- Expected frequency ratios: f2/f1 ≈ 4.0, f3/f1 ≈ 9.0 (vs 6.27, 17.5 cantilever)
- DOF removal: constrain nodes 0 and n (vertical displacement only)

The Z24 bridge is a 30m pre-stressed concrete box girder that underwent
controlled progressive damage (pier settlement) before demolition in 1998.
This model provides the physics-based forward model for PINN-SHM validation.
"""

import torch
import torch.nn as nn

try:
    from ..pinn_shm.models.beam_model import BeamModel
    from ..pinn_shm.config import DTYPE
except ImportError:
    # Allow running from shm/z24/ directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pinn_shm.models.beam_model import BeamModel
    from pinn_shm.config import DTYPE


class Z24BridgeModel(BeamModel):
    """Simply-supported beam model for Z24 bridge.

    The Z24 bridge is a 30m pre-stressed concrete box girder with pinned supports
    at both ends. This model adapts BeamModel's cantilever formulation to handle
    simply-supported boundary conditions.

    Boundary Conditions:
    - Left support (node 0): Fixed vertical displacement (DOF 0 = 0),
                            free rotation (DOF 1 ≠ 0)
    - Right support (node n): Fixed vertical displacement (DOF 2n = 0),
                             free rotation (DOF 2n+1 ≠ 0)

    For comparison, cantilever beams fix BOTH displacement and rotation at base:
    - Fixed end (node 0): DOF 0 = 0, DOF 1 = 0
    - Free end (node n): all DOFs free

    Args:
        length: Bridge span in metres (default 30.0 for Z24)
        n_elements: Number of finite elements (default 15)
        rho_a: Mass per unit length (arbitrary normalisation, default 1.0)
        n_modes: Number of natural frequencies to extract (default 5)

    Example:
        >>> model = Z24BridgeModel(length=30.0, n_elements=15, n_modes=5)
        >>> model.calibrate_ei(target_freq_hz=4.0)
        >>> theta_healthy = torch.zeros(15, dtype=DTYPE)
        >>> freqs = model.forward(theta_healthy)
        >>> print(f"f1 = {freqs[0]:.3f} Hz")  # Should print ~4.0 Hz
    """

    def __init__(self, length=30.0, n_elements=15, rho_a=1.0, n_modes=5):
        # Initialize parent (BeamModel)
        # This sets up element templates, assembly buffers, and caches M_bc for cantilever
        super().__init__(length, n_elements, rho_a, n_modes)

        # Recompute boundary condition buffers for simply-supported
        # Parent initialized _M_bc for cantilever ([2:, 2:])
        # We need to remove DOFs [0, 2*n_elements] instead
        self._initialize_simply_supported_bc()

    def _initialize_simply_supported_bc(self):
        """Recalculate mass matrix caches for simply-supported BC.

        Simply-supported beams fix vertical displacement at both ends:
        - DOF 0: vertical displacement at left support (node 0)
        - DOF 2*n_elements: vertical displacement at right support (node n)

        All rotation DOFs remain free.

        This method:
        1. Builds a list of free DOFs (all except 0 and 2*n_elements)
        2. Extracts the reduced mass matrix M_bc from global M
        3. Computes Cholesky decomposition of M_bc (needed by physics loss)
        4. Caches all three as buffers for reuse

        Note: The parent BeamModel.__init__() already cached M_bc for cantilever
        ([2:, 2:] slicing). We override those buffers here.
        """
        # Build free DOF mask
        # For n_elements=15: total DOFs = 2*(15+1) = 32
        # Remove DOF 0 (left vertical) and DOF 30 (right vertical)
        # Free DOFs: [1, 2, 3, ..., 29, 31] → 30 free DOFs
        n_dofs = self.n_dofs
        free_dofs = []

        for i in range(n_dofs):
            # Skip DOF 0 (left vertical displacement)
            if i == 0:
                continue
            # Skip DOF 2*n_elements (right vertical displacement)
            # For n_elements=15: skip DOF 30
            if i == 2 * self.n_elements:
                continue
            free_dofs.append(i)

        # Convert to tensor for advanced indexing
        free_dofs_tensor = torch.tensor(free_dofs, dtype=torch.long)

        # Extract simply-supported mass matrix from global
        # Uses fancy indexing: M_ss = M[free_dofs][:, free_dofs]
        # Equivalent to: M_ss[i,j] = M_global[free_dofs[i], free_dofs[j]]
        M_ss = self._M_global[free_dofs_tensor][:, free_dofs_tensor]

        # Replace cached mass matrix (overwrite parent's cantilever version)
        # This buffer is used by apply_bc() when M=None
        self.register_buffer('_M_bc', M_ss.clone())

        # Recalculate Cholesky decomposition (needed by physics loss)
        # L @ L.T = M_bc
        L = torch.linalg.cholesky(self._M_bc)
        self.register_buffer('_L_cholesky', L, persistent=False)

        # Precompute L^{-T} = (L^{-1})^T for eigenvalue transformation
        # Used in PINNLoss to transform eigenvectors: phi = L^{-T} @ psi
        self.register_buffer('_L_inv_T', torch.linalg.inv(L).T, persistent=False)

    def apply_bc(self, K, M=None):
        """Apply simply-supported boundary conditions.

        Removes DOFs corresponding to vertical displacement at both ends:
        - DOF 0: left support
        - DOF 2*n_elements: right support

        Keeps all rotation DOFs free (DOFs 1, 3, 5, ..., 2n-1, 2n+1).

        Args:
            K: Global stiffness matrix, shape (n_dofs, n_dofs)
            M: Global mass matrix, shape (n_dofs, n_dofs) or None
               If None, uses cached _M_bc from __init__

        Returns:
            K_bc: Reduced stiffness matrix, shape (n_free, n_free)
            M_bc: Reduced mass matrix (cached if M=None), shape (n_free, n_free)

        Example:
            For n_elements=15:
            - Input: K is 32×32, M is 32×32
            - Output: K_bc is 30×30, M_bc is 30×30
            - Removed: DOFs [0, 30]
        """
        # Build free DOF list (same logic as _initialize_simply_supported_bc)
        # This is duplicated rather than cached to handle arbitrary K device
        n_dofs = self.n_dofs
        free_dofs = []

        for i in range(n_dofs):
            if i == 0 or i == 2 * self.n_elements:
                continue
            free_dofs.append(i)

        free_dofs_tensor = torch.tensor(free_dofs, dtype=torch.long, device=K.device)

        # Extract reduced matrices using advanced indexing
        K_bc = K[free_dofs_tensor][:, free_dofs_tensor]

        if M is None:
            # Use cached mass matrix (already reduced)
            M_bc = self._M_bc
        else:
            # Reduce provided mass matrix
            M_bc = M[free_dofs_tensor][:, free_dofs_tensor]

        return K_bc, M_bc
