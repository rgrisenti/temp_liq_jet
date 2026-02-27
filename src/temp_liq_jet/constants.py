from scipy.constants import Avogadro

LIQUID_PROPERTIES = {
    "water": {
        "molar_mass": 18.015e-3, # (Kg/mol)
        "atomic_mass": 18.015e-3/Avogadro, # (Kg)
        "T_m": 273.15 # (K)
    },
    "argon": {
        "molar_mass": 39.948e-3,
        "atomic_mass": 39.948e-3/Avogadro,
        "T_m": 83.8
    },
    "krypton": {
        "molar_mass": 83.798e-3,
        "atomic_mass": 83.798e-3/Avogadro,
        "T_m": 115.8
    }
}