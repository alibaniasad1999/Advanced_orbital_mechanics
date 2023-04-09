import numpy as np

def ecef2latlon(r):
    """
    Convert ECEF coordinates to latitude and longitude.
    """

    r_k = r[-1]

    delta = np.arcsin(r_k / np.linalg.norm(r))

    ## earth data ##
    R_earth = 6378.1370 # km
    e_earth = 0.0818191908426

    phi_new = delta
    phi_old = np.inf
    while np.abs(phi_new - phi_old) > 1e-10:
        phi_old = phi_new
        C_earth = R_earth / (np.sqrt(1 - e_earth**2 * np.sin(delta)**2))
        phi_new = np.arctan((r_k + C_earth * e_earth**2 * np.sin(delta)) / np.linalg.norm(r[0:2]))
    
    h_ellp = np.linalg.norm(r[0:2]) / np.cos(phi_new) - C_earth

    landa = np.arcsin(r[1] / (np.linalg.norm(r[0:2])))
    
    return phi_new, landa, h_ellp