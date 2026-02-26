"""
Knudsen jet evaporation model.

This module implements the KnudsenModel class and supporting thermodynamic
functions to simulate the evaporation and cooling of liquid jets (water,
argon, krypton). The model solves coupled mass and heat transport equations
for concentric shells using SciPy ODE solvers.

Main class
----------
KnudsenModel : simulate the jet dynamics and provide interpolated profiles
               for shell temperatures, jet radius, and average/core/surface
               temperatures.

Supporting functions
--------------------
rho        : Return liquid density as a function of temperature.
c_p        : Return specific heat capacity.
p_sat      : Return saturated vapor pressure.
L_v        : Return enthalpy of vaporization.
therm_cond : Return thermal conductivity.
"""

import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, make_interp_spline
import warnings

# ----- Some constants ---------------------
# Boltzmann's constant
k_B = 1.38066e-23 

# ----- Define liquid properties -----------
# Each liquid is defined by its molar mass (kg/mol),
# atomic mass (kg), and melting temperature (K).

LIQUID_PROPERTIES = {
    "water": {
        "molar_mass": 18.015e-3, # (Kg/mol)
        "atomic_mass": 18.015e-3/6.022e23, # (Kg)
        "T_m": 273.15
    },
    "argon": {
        "molar_mass": 39.948e-3,
        "atomic_mass": 39.948e-3/6.022e23,
        "T_m": 83.8
    },
    "krypton": {
        "molar_mass": 83.798e-3,
        "atomic_mass": 83.798e-3/6.022e23,
        "T_m": 115.8
    }
}

# ----- Define thermodynamic functions -----
def rho(liquid, w_rho_model, T):
    """
    Return the density of the liquid at temperature T.

    Parameters
    ----------
    liquid      : {'argon', 'krypton', 'water'}
                  Name of the liquid.
    w_rho_model : {'Hare1987', 'Caupin2019'}
                  Specifies the model for the density of supercooled liquid water.
    T           : float or ndarray
                  Temperature in (K).

    Returns
    -------
    rho : float or ndarray
          Density in (kg/m^3).
    """
    if liquid == 'argon':
        rT = 1 - T/150.687
        exponent = (1.5*rT**0.334
                    - 0.314*rT**(2/3)
                    + 0.086*rT**(7/3)
                    - 0.041*rT**4)
        return 535.599*np.exp(exponent)

    elif liquid == 'krypton':
        rT = 1 - T/209.48
        return 909.2*(1 + 2.422741*rT**0.448)

    elif liquid == 'water':
        if w_rho_model == 'Hare1987':
            return (-671649.56609929 + 14584.455054882*T
                    - 131.95828285732*T**2
                    + 0.63749917015289*T**3
                    - 0.001733846814305*T**4
                    + 2.516524948219e-6*T**5
                    - 1.5225226829696e-9*T**6)

        elif w_rho_model == 'Caupin2019':
            T = np.asarray(T)  # Ensures array-like operations
            out = np.empty_like(T, dtype = float)

            mask_low = T < 239
            mask_high = ~mask_low

            # Apply polynomial 1 only to low-T part
            out[mask_low] = (33169.4610493092 + 7162.69779018419*(T[mask_low] - 273.15)
                            + 692.242318914883*(T[mask_low] - 273.15)**2
                            + 37.9196754533035*(T[mask_low] - 273.15)**3
                            + 1.2871452602957*(T[mask_low] - 273.15)**4
                            + 0.0277132359010566*(T[mask_low] - 273.15)**5
                            + 3.69466713035065e-4*(T[mask_low] - 273.15)**6
                            + 2.78777027355768e-6*(T[mask_low] - 273.15)**7
                            + 9.11410008186836e-9*(T[mask_low] - 273.15)**8)

            # Apply polynomial 2 only to high-T part
            out[mask_high] = (-671649.56609929 + 14584.455054882*T[mask_high]
                              - 131.95828285732*T[mask_high]**2
                              + 0.63749917015289*T[mask_high]**3
                              - 0.001733846814305*T[mask_high]**4
                              + 2.516524948219E-6*T[mask_high]**5
                              - 1.5225226829696E-9*T[mask_high]**6)
            return out


def c_p(liquid, w_cp_model, T):
    """
    Return the specific heat capacity of the liquid at temperature T.

    Parameters
    ----------
    liquid      : {'argon', 'krypton', 'water'}
                  Name of the liquid.
    w_cp_model  : {'Angell1982', 'Archer2000', 'Pathak2021'}
                  Specifies which experimental dataset is used to parametrize the analytic
                  exponential expression for the specific heat capacity of supercooled water.
    T           : float or ndarray
                  Temperature in (K).

    Returns
    -------
    c_p : float or ndarray
          Specific heat capacity in (J/(kg K)).
    """
    if liquid == 'argon':
        m_mol = LIQUID_PROPERTIES[liquid]['molar_mass']
        return (28.6384 + 0.1744*T + 4.66e-08*T*np.exp(0.111*T))/m_mol

    elif liquid == 'krypton':
        m_mol = LIQUID_PROPERTIES[liquid]['molar_mass']
        return (40.5517 + 0.0349*T + 1.7169e-06*T*np.exp(0.0599*T))/m_mol

    elif liquid == 'water':
        if w_cp_model == 'Angell1982':
            """
            Fit to exp. data from Angell et al. (1982).
            """
            return 4186.93717545 + 5425.84045584*np.exp(-(T - 224.48917491)/8.86275377)

        elif w_cp_model == 'Archer2000':
            """
            Fit to exp. data from Archer and Carter (2000).
            """
            return 4197.14238779 + 4595.88383033*np.exp(-(T - 224.41061552)/8.61062621)

        elif w_cp_model == 'Pathak2021':
            """
            Fit to exp. data from Pathak et al. (2021).
            """
            return 4186.93717545 + 4148.09068582*np.exp(-(T - 232.18766899)/5.57600728)

def p_sat(liquid, w_cp_model, T):
    """
    Return the saturated vapor pressure of the liquid at temperature T.

    Parameters
    ----------
    liquid      : {'argon', 'krypton', 'water'}
                  Name of the liquid.
    T           : float or ndarray
                  Temperature in (K).

    Returns
    -------
    p_sat : float or ndarray
            Saturated vapor pressure in (Pa = kg/(m s^2)).
    """
    if liquid == 'argon':
        T_c = 150.687
        rT = 1 - T/T_c
        return 4.863e06*np.exp((T_c/T)*(-5.9409785*rT + 1.3553888*rT**1.5 - 0.46497608*rT**2 - 1.5399043*rT**4.5))

    elif liquid == 'krypton':
        T_c = 209.48
        rT = 1 - T/T_c
        return 5.5e06*np.exp((T_c/T)*(-5.8964*rT + 1.0783*rT**1.5 - 0.2442*rT**2.5 - 2.4242*rT**5))

    elif liquid == 'water':
        if w_cp_model == 'Angell1982' or w_cp_model == 'Pathak2021':
            """
            IAPWS-95 parametrization.
            """
            T_c = 647.096
            rT = 1 - T/T_c
            log_psat = (T_c/T)*(
                                - 7.85951783*rT + 1.84408259*rT**1.5 - 11.7866497*rT**3
                                + 22.6807411*rT**3.5 - 15.9618719*rT**4 + 1.80122502*rT**7.5
                                )

            return 22.064e06*np.exp(log_psat)

        elif w_cp_model == 'Archer2000':
            """
            Eq. (10) from Murphy and Koop, "Review of the vapour pressures of
            ice and supercooled water for atmospheric applications" (2005).
            """
            log_psat = (54.842763
                        - 6763.22/T
                        - 4.210*np.log(T)
                        + 0.000367*T
                        + np.tanh(0.0415*(T - 218.8))*(
                                                       53.878
                                                       - 1331.22/T
                                                       - 9.44523*np.log(T)
                                                       + 0.014025*T
                                                       )
            )

            return np.exp(log_psat)

def L_v(liquid, w_cp_model, T):
    """
    Return the enthalpy of vaporization of the liquid at temperature T
    computed analytically from the Clausius–Clapeyron equation.

    Parameters
    ----------
    liquid      : {'argon', 'krypton', 'water'}
                  Name of the liquid.
    T           : float or ndarray
                  Temperature in (K).

    Returns
    -------
    L_v : float or ndarray
          Enthalpy of vaporization in (J/kg).
    """
    # Specific gas constant
    R_s = 8.314/LIQUID_PROPERTIES[liquid]['molar_mass']

    if liquid == 'argon':
        T_c = 150.687
        rT = 1 - T/T_c

        # F(rT)
        F = (
             - 5.9409785*rT
             + 1.3553888*rT**1.5
             - 0.46497608*rT**2
             - 1.5399043*rT**4.5
        )

        # dF/drT
        dF_drT = (
                  - 5.9409785
                  + 1.5*1.3553888*rT**0.5
                  - 2.0*0.46497608*rT
                  - 4.5*1.5399043*rT**3.5
        )
        return - R_s*(T_c*F + T*dF_drT)

    elif liquid == 'krypton':
        T_c = 209.48
        rT = 1 - T/T_c

        # F(rT)
        F = (
             - 5.8964*rT
             + 1.0783*rT**1.5
             - 0.2442*rT**2.5
             - 2.4242*rT**5
        )

        # dF/drT
        dF_drT = (
                  - 5.8964
                  + 1.5*1.0783*rT**0.5
                  - 2.5*0.2442*rT**1.5
                  - 5.0*2.4242*rT**4
        )
        return - R_s*(T_c*F + T*dF_drT)

    elif liquid == 'water':
        if w_cp_model == 'Angell1982' or w_cp_model == 'Pathak2021':
            """
            Latent heat of vaporization computed from the Clausius–Clapeyron equation
            using the IAPWS-95 saturation vapor pressure formulation.
            """
            T_c = 647.096
            rT = 1 - T/T_c
            F = (
                 - 7.85951783*rT
                 + 1.84408259*rT**1.5
                 - 11.7866497*rT**3
                 + 22.6807411*rT**3.5
                 - 15.9618719*rT**4
                 + 1.80122502*rT**7.5
            )

            # dF/dr
            dF_dr = (
                     - 7.85951783
                     + 1.5*1.84408259*rT**0.5
                     - 3.0*11.7866497*rT**2
                     + 3.5*22.6807411*rT**2.5
                     - 4.0*15.9618719*rT**3
                     + 7.5*1.80122502*rT**6.5
            )
            return - R_s*(T_c*F + T*dF_dr)

        elif w_cp_model == 'Archer2000':
            """
            Latent heat of vaporization computed from the Clausius–Clapeyron equation
            using the saturated vapor pressure formula of Murphy and Koop (2005).
            """
            # Original functions
            F2 = 53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T
            tanh_term = np.tanh(0.0415*(T - 218.8))

            # Derivatives
            dF1_dT = 6763.22/T**2 - 4.210/T + 0.000367
            dF2_dT = 1331.22/T**2 - 9.44523/T + 0.014025

            dlnp_dT = dF1_dT + 0.0415*(1 - tanh_term**2)*F2 + tanh_term*dF2_dT

            return R_s*(T**2)*dlnp_dT

def therm_cond(liquid, T):
    """
    Return the thermal conductivity of the liquid at temperature T.

    Parameters
    ----------
    liquid      : {'argon', 'krypton', 'water'}
                  Name of the liquid.
    T           : float or ndarray
                  Temperature in (K).

    Returns
    -------
    therm_cond : float or ndarray
                 Thermal conductivity in (W/(m K)).
    """
    if liquid == 'argon':
        T_m = LIQUID_PROPERTIES[liquid]['T_m']
        return 0.24032 - 0.00131*T_m + 0.00131*(T_m - T)

    elif liquid == 'krypton':
        T_m = LIQUID_PROPERTIES[liquid]['T_m']
        return 0.21793 - 0.00089714*T_m + 0.00089714*(T_m - T)

    elif liquid == 'water':
        rT = T/300
        return 1.663/rT**1.15 - 1.7781/rT**3.4 + 1.1567/rT**6 - 0.432115/rT**7.6

# ------------------------------ Define main class -------------------------------------------
class KnudsenModel:
    """
    Knudsen jet evaporation model.

    Simulates the evaporation and thermal evolution of a liquid jet
    represented by concentric shells. The governing equations are integrated
    using SciPy's ODE solver (`solve_ivp`), with adaptive handling when the
    outer shell evaporates.

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
