import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, make_interp_spline
from scipy.constants import Boltzmann as k_B
import warnings

from .constants import LIQUID_PROPERTIES
from .thermodynamics import rho, c_p, p_sat, L_v, therm_cond

class KnudsenModel:
    """
    Knudsen jet evaporative cooling model.

    This module implements the KnudsenModel class to compute the evaporative cooling
    of liquid jets (water, argon, krypton). The model solves coupled mass and heat
    transport equations for concentric shells using SciPy ODE solvers.

    Parameters
    ----------
    liquid      : {'water', 'argon', 'krypton'}
                  Name of the liquid.
    w_cp_model  : {'Angell1982', 'Archer2000', 'Pathak2021'}
                  Heat capacity model identifier (only used for water).
    w_rho_model : {'Hare1987', 'Caupin2019'}
                  Density model identifier (only used for water).
    D           : int
                  Dimensionality of the jet (2 for cylinder, 3 for sphere).
    T_nozzle    : float
                  Nozzle temperature (K).
    d           : float
                  Initial jet/droplet diameter (m).
    v           : float
                  Jet velocity (m/s).
    N           : int
                  Number of concentric shells.
    z_end_mm    : float
                  Integration length in millimeters.
    p_amb       : float
                  Ambient pressure (mbar). Default is 0 mbar.
    evap_coef   : float
                  Evaporation coefficient. Default is 1.
    accuracy    : {'low', 'medium', 'high'}
                  Define mapping from accuracy level to solver tolerances. Default is 'medium'.

    Attributes
    ----------
    z_all : list of ndarray
            List of distance arrays (per evaporation stage).
    y_all : list of ndarray
            List of state arrays [mass, temperatures] (per stage).
    m0    : float
            Initial mass per shell.
    """
    def __init__(self, liquid, w_cp_model, w_rho_model, D, T_nozzle, d, v, N, z_end_mm, *, p_amb = 0, evap_coef = 1, accuracy = "medium"):
        # ----- Safety checks ---------
        if liquid not in LIQUID_PROPERTIES:
            raise ValueError(f"Invalid liquid '{liquid}'. Must be one of {list(LIQUID_PROPERTIES)}.")
    
        if liquid == "water" and w_cp_model not in {"Angell1982", "Archer2000", "Pathak2021"}:
            raise ValueError("For water, w_cp_model must be one of {'Angell1982', 'Archer2000', 'Pathak2021'}.")

        if liquid != "water" and w_cp_model is not None:
            raise ValueError(f"w_cp_model is only used for water, but liquid = '{liquid}' was given.")

        if liquid == "water" and w_rho_model not in {"Hare1987", "Caupin2019"}:
            raise ValueError("For water, w_rho_model must be one of {'Hare1987', 'Caupin2019'}.")

        if liquid != "water" and w_rho_model is not None:
            raise ValueError(f"w_rho_model is only used for water, but liquid = '{liquid}' was given.")

        if D not in (2, 3):
            raise ValueError("D must be 2 (cylinder) or 3 (sphere).")

        if T_nozzle <= 0:
            raise ValueError("Nozzle temperature T_nozzle must be > 0 K.")
        if d <= 0:
            raise ValueError("Initial diameter d must be > 0 m.")
        if v <= 0:
            raise ValueError("Jet velocity v must be > 0 m/s.")
        if N < 10:
            raise ValueError("Number of shells N must be at least 10.")
        if z_end_mm <= 0:
            raise ValueError("Integration length z_end_mm must be > 0.")
        if p_amb < 0 or p_amb > 1:
            raise ValueError("Ambient pressure must be between 0 and 1 mbar (inclusive).")
        if evap_coef <= 0 or evap_coef > 1:
            raise ValueError("Evaporation coefficient must be > 0 and <= 1.")

        accuracy_map = {
            "low":    {"rtol": 1e-3, "atol": 1e-6},
            "medium": {"rtol": 1e-6, "atol": 1e-9},
            "high":   {"rtol": 1e-9, "atol": 1e-12}
        }

        if accuracy not in accuracy_map:
            raise ValueError(f"Invalid accuracy level '{accuracy}'. Choose from 'low', 'medium', 'high'.")

        # ----- Assign attributes -----
        self.liquid = liquid
        self.w_cp_model = w_cp_model
        self.w_rho_model = w_rho_model
        self.D = D
        self.T_nozzle = T_nozzle
        self.d = d
        self.v = v
        self.N = N
        self.z_end_mm = z_end_mm
        self.p_amb = p_amb
        self.evap_coef = evap_coef
        self.rtol = accuracy_map[accuracy]["rtol"]
        self.atol = accuracy_map[accuracy]["atol"]
        
        # Initialize results
        self.z_all = []
        self.y_all = []

        # Run simulation
        self._run_simulation()

    def _run_simulation(self):
        """
        Run the full jet evaporation simulation.

        Integrates the system of ODEs.
        Stores results in `z_all` and `y_all`.
        """
        
        # ------------------------------ Define functions first ------------------------------

        # ----- Reduced radius of the l-th shell, l>=1 ----------------------
        def rD_l(l, density):
            return np.sum(1/density[l:])
        
        # ----- Differential equations --------------------------------------
        def dydz(z, y):

            dy = np.zeros_like(y)

            # Mass of the outer shell in reduced units (relative to m0)
            m_0 = y[0]
            # Shell temperatures
            T = y[1:]

            # Compute the thermodynamic functions only once
            c_p_T = c_p(self.liquid, self.w_cp_model, T)
            l_T = therm_cond(self.liquid, T)
            L_v_T0 = L_v(self.liquid, self.w_cp_model, T[0])
            p_sat_T0 = p_sat(self.liquid, self.w_cp_model, T[0])
            rho_T = rho(self.liquid, self.w_rho_model, T)

            rD_1 = rD_l(1, rho_T)
            r_1 = (fact*rD_1)**(1/self.D)

            # Jet/droplet radius
            r_0 = (
                   fact
                   *(m_0/rho_T[0] + rD_1)
            )**(1/self.D)

            if self.D == 2:
                alpha_0 = 1/math.log(r_0/r_1)
            elif self.D == 3:
                alpha_0 = 1/(1/r_1 - 1/r_0)

            # Mass loss of the outer shell
            dy[0] = - (
                       self.evap_coef
                       *fact2
                       *r_0**(self.D - 1)
                       *(p_sat_T0 - self.p_amb*100)
                       *np.sqrt(m/(2*math.pi*k_B*T[0]))
            )

            # Temperature variation of the outer shell
            dy[1] = (
                     L_v_T0/c_p_T[0]
                     *dy[0]/m_0
                     - fact2
                     *1/c_p_T[0]
                     *alpha_0*l_T[0]/m_0
                     *(T[0] - T[1])
            )
            
            # Temperature variations of the next N-1 inner shells
            for k in range(1, N - 1):

                rD_k = fact*rD_l(k, rho_T)
                rD_kplus1 = rD_k - fact/rho_T[k]
                rD_kmin1 = rD_k + fact/rho_T[k - 1]

                if k == 1:
                    alpha_kmin1 = alpha_0
                elif self.D == 2:
                    alpha_kmin1 = 1/math.log(math.sqrt(rD_kmin1/rD_k))
                elif self.D == 3:
                    alpha_kmin1 = 1/(1/(rD_k**(1/3)) - 1/rD_kmin1**(1/3))

                if self.D == 2:
                    alpha_k = 1/math.log(math.sqrt(rD_k/rD_kplus1))
                elif self.D == 3:
                    alpha_k = 1/(1/rD_kplus1**(1/3) - 1/rD_k**(1/3))
                
                dy[k + 1] = - (
                               fact2
                               *1/c_p_T[k]
                               *(
                                 alpha_k*l_T[k]*(T[k] - T[k + 1])
                                 - alpha_kmin1*l_T[k - 1]*(T[k - 1] - T[k])
                               )
                )

            # Temperature variation of the jet's core
            if self.D == 2:
                alpha_Nmin2 = 1/math.log(math.sqrt(rD_l(N - 2, rho_T)/rD_l(N - 1, rho_T)))
            elif self.D == 3:
                alpha_Nmin2 = 1/(1/(fact*rD_l(N - 1, rho_T))**(1/3) - 1/(fact*rD_l(N - 2, rho_T))**(1/3))

            dy[-1] = (
                      fact2
                      *1/c_p_T[-1]
                      *alpha_Nmin2*l_T[-2]*(T[-2] - T[-1])
            )

            return dy

        # ----- Define an event when the outer shell mass drops to zero -----
        def event_evaporate(z, y):
            return y[0]

        event_evaporate.terminal = True
        event_evaporate.direction = -1

        # ---------------------------------------------------------------------------------

        N = self.N
        # Conversion in (m)
        z_end = self.z_end_mm/1000

        m = LIQUID_PROPERTIES[self.liquid]['atomic_mass']
        
        # Initial shell mass
        self.m0 = (
                   (4/3)**(self.D - 2)*math.pi
                   *rho(self.liquid, self.w_rho_model, self.T_nozzle)
                   *(self.d/2)**self.D
        )/self.N

        fact = self.m0/((4/3)**(self.D - 2)*math.pi)
        fact2 = 2**(self.D - 1)*math.pi/(self.m0*self.v)
        
        y0 = np.zeros(N + 1)
        y0[0] = 1
        y0[1:] = self.T_nozzle
        
        z_current = 0
        status = 1

        while status > 0:
            try:
                with warnings.catch_warnings():
                    # Promote numerical RuntimeWarnings into exceptions
                    warnings.filterwarnings("error", category = RuntimeWarning)
            
                    solution = solve_ivp(
                                dydz, (z_current, z_end), y0,
                                events = event_evaporate,
                                method = 'LSODA',
                                rtol = self.rtol,
                                atol = self.atol)

                # Create lists of arrays
                self.z_all.append(solution.t)
                self.y_all.append(solution.y)
                
                status = solution.status # Check if an event took place (mass dropped to zero)

                if status > 0:                         
                    z_current = solution.t_events[0][0] # Distance at which the shell mass dropped to zero
                        
                    N = N - 1 # The number of shells decreases by one unity
                    y0 = np.zeros(N + 1)
                        
                    # The new initial conditions correspond to the output from the previous iteration
                    y0[0] = 1
                    y0[1:] = solution.y[2:, -1]

            except RuntimeWarning as e:
                raise RuntimeError(
                    "Numerical instability detected during integration. "
                    "Consider reducing N or increasing accuracy (e.g., set accuracy = 'high')."
                ) from e

    # ----- Methods -----
    def norm_mass(self):
        """
        Return distance and corresponding normalized mass of outermost shell.

        Returns
        -------
        z_all         : list of arrays
                        Contains the distances at which the
                        normalized outer shell mass is computed.
        norm_mass_all : list of arrays
                        Contains the normalized mass of the outer shell.
        """

        z_all = []
        norm_mass_all = []

        for z, y in zip(self.z_all, self.y_all):
            z_all.append(z)
            norm_mass_all.append(y[0, :])
         
        return z_all, norm_mass_all

    def temperature(self):
        """
        Return distance ranges and corresponding interpolated shell temperatures.

        Returns
        -------
        z_ranges_all : list of tuple of float
                       Each tuple contains the lower and upper distance bounds
                       where the outer shell mass is finite.
        T_shell_all  : list of callable
                       List of interpolation functions (splines) giving the
                       temperature of each shell as a function of distance.
        """

        z_range_all = []
        T_shell_all = []

        for z, y in zip(self.z_all, self.y_all):
            
            z_range_all.append([z[0], z[-1]])

            T_shell = []
            for i in range(1, y.shape[0]):
                T_shell.append(make_interp_spline(z, y[i, :]))
        
            T_shell_all.append(T_shell)

        return z_range_all, T_shell_all

    def avg_temperature(self):
        """
        Return interpolated, mass-weighted average jet temperature.

        Returns
        -------
        spline : callable
                 Interpolation function (B-spline) mapping distance `z` (m)
                 to the average jet temperature (K).
        """
        avg_temp_combined = []

        for y in self.y_all:
            T_avg = (
                (
                    y[0, :]*y[1, :] + np.sum(y[2:, :], axis = 0)
                )
                /(
                    y[0, :] + y[2:, :].shape[0]
                )
            )
            avg_temp_combined.append(T_avg)

        avg_temp_combined = np.concatenate(avg_temp_combined)
        z_combined = np.concatenate(self.z_all)

        z_unique, idx = np.unique(z_combined, return_index = True)
        avg_temp_unique = avg_temp_combined[idx]

        return make_interp_spline(z_unique, avg_temp_unique)

    def surface_temp(self):
        """
        Return interpolated outermost shell temperature.

        Returns
        -------
        spline : callable
                 Interpolation function (B-spline) mapping distance `z` (m)
                 to the surface shell temperature (K).
        """
        surface_temp_combined = []

        for y in self.y_all:
            surface_temp_combined.append(y[1, :])

        surface_temp_combined = np.concatenate(surface_temp_combined)
        z_combined = np.concatenate(self.z_all)

        z_unique, idx = np.unique(z_combined, return_index = True)
        surface_temp_unique = surface_temp_combined[idx]

        return make_interp_spline(z_unique, surface_temp_unique)

    def core_temp(self):
        """
        Return interpolated innermost shell temperature.

        Returns
        -------
        spline : callable
                 Interpolation function (B-spline) mapping distance `z` (m)
                 to the core (innermost shell) temperature (K).
        """
        core_temp_combined = []

        for y in self.y_all:
            core_temp_combined.append(y[-1, :])

        core_temp_combined = np.concatenate(core_temp_combined)
        z_combined = np.concatenate(self.z_all)

        z_unique, idx = np.unique(z_combined, return_index = True)
        core_temp_unique = core_temp_combined[idx]

        return make_interp_spline(z_unique, core_temp_unique)

    def radius(self):
        """
        Return interpolated jet radius.

        Returns
        -------
        spline : callable
                 Interpolation function (B-spline) mapping distance `z` (m)
                 to the jet (droplet) radius (m).
        """
        radius_combined = []

        for y in self.y_all:
            r_to_D = (
                      self.m0/((4/3)**(self.D - 2)*math.pi)
                      *(
                        y[0, :]/rho(self.liquid, self.w_rho_model, y[1, :])
                        + np.sum(1/rho(self.liquid, self.w_rho_model, y[2:, :]), axis = 0)
                    )
            )
            r = r_to_D**(1/self.D)
            radius_combined.append(r)

        radius_combined = np.concatenate(radius_combined)
        z_combined = np.concatenate(self.z_all)

        z_unique, idx = np.unique(z_combined, return_index = True)
        radius_unique = radius_combined[idx]

        return make_interp_spline(z_unique, radius_unique)
