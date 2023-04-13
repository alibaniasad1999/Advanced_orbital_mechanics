from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body_barycentric, get_body_barycentric_posvel
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body_barycentric, get_body_barycentric_posvel
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_body, SkyCoord

from astropy.coordinates import TEME, CartesianDifferential, CartesianRepresentation
from astropy import units as u
# r0 = orb.r.value
# v0 = orb.v.value
# teme_p = CartesianRepresentation(r0*u.km)
# teme_v = CartesianDifferential(v0*u.km/u.s)
# t = Time(2458827.362605, format='jd')
# teme = TEME(teme_p.with_differentials(teme_v), obstime=t)
## function check ground station visibility
lat, lon, elev = 34.224694, -118.057306, 0.03  # Latitude, longitude, and elevation of Los Angeles
los_angeles = EarthLocation(lat=lat, lon=lon, height=elev)
utc_time = '2000-01-01T12:00:00'
t = Time(utc_time, format='isot', scale='utc')
def check_visibility(pos, los_angeles, t):
    # Convert the position of the ISS to astropy coordinates
    iss_pos = SkyCoord(pos[0], pos[1], pos[2], unit='km', frame='gcrs', obstime=t, representation_type='cartesian')
    # Convert the position of the ISS to AltAz coordinates
    iss_pos_altaz = iss_pos.transform_to(AltAz(obstime=t, location=los_angeles))
    # Check if the ISS is above the horizon
    iss_above_horizon = iss_pos_altaz.alt > 0*u.deg
    # Check if the ISS is above the elevation limit
    iss_above_elevation_limit = iss_pos_altaz.alt > 10*u.deg
    # Check if the ISS is above the elevation limit and above the horizon
    iss_above_elevation_limit_and_above_horizon = iss_above_elevation_limit & iss_above_horizon
    if iss_above_elevation_limit_and_above_horizon:
        return 1
    else:
        return 0
    
result = 0
print(check_visibility(ISS, los_angeles, t))
for i in range(2*24*36):
    t = Time(utc_time, format='isot', scale='utc') + timedelta(seconds=i)
    meysa = check_visibility(state[i, 0:3], los_angeles, t)
    print(result)
    result += meysa