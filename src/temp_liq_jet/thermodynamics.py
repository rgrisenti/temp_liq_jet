import numpy as np
from .constants import LIQUID_PROPERTIES

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
        """
        For water:
        - 'Caupin2019' is used for T >= 273.15 K, which is the only valid model above melting.
        - If w_rho_model='Hare1987', Hare1987 is used only for T < 273.15 K.
        - Mixed temperature arrays are handled element-wise.
        """
        # Ensures array-like operations
        T = np.asarray(T)

        density_C = np.empty_like(T, dtype = float)

        mask_low = T < 239
        mask_high = ~mask_low
        # Apply polynomial 1 only to low-T part
        T_low = T[mask_low] - 273.15 # Conversion to (°C)
        density_C[mask_low] = (33169.4610493092 + 7162.69779018419*T_low
                               + 692.242318914883*T_low**2
                               + 37.9196754533035*T_low**3
                               + 1.2871452602957*T_low**4
                               + 0.0277132359010566*T_low**5
                               + 3.69466713035065e-4*T_low**6
                               + 2.78777027355768e-6*T_low**7
                               + 9.11410008186836e-9*T_low**8)
        # Apply polynomial 2 only to high-T part
        density_C[mask_high] = (-671649.56609929 + 14584.455054882*T[mask_high]
                                - 131.95828285732*T[mask_high]**2
                                + 0.63749917015289*T[mask_high]**3
                                - 0.001733846814305*T[mask_high]**4
                                + 2.516524948219e-6*T[mask_high]**5
                                - 1.5225226829696e-9*T[mask_high]**6)

        
        # Now check whether the user has chosen the 'Hare1987' parametrization
        if w_rho_model == 'Hare1987':
            t = T - 273.15 # Conversion to (°C)
            # Density in (g/ml)
            density_H = (0.99986
                         + 6.69e-05*t
                         - 8.484e-06*t**2
                         + 1.518e-07*t**3
                         - 6.9484e-09*t**4
                         - 3.6449e-10*t**5
                         - 7.497e-12*t**6)

            density_H = density_H*1000 # Conversion to (kg/m^3)
            density = np.where(T >= 273.15, density_C, density_H)
            return density.item() if density.ndim == 0 else density

        elif w_rho_model == 'Caupin2019':
            return density_C.item() if density_C.ndim == 0 else density_C


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