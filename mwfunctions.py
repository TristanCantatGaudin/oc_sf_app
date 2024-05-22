import astropy.coordinates as coord
import astropy.units as u
import numpy as np

def xyz_to_pmradec(x, y, z, logt=9., factor=1.):
    """
    Input:
        x, y, z: Galactocentric cartesian coordinates, in kpc
        logt: age, in log(yr)
        factor: multiply the velocity dispersions by a factor
        
    Output:
        pmra
        pmdec
        all_vR
        all_vPhi
        all_vZ
        
    Assumptions:
        * default astropy Solar location and motion (notably, Rsun=8kpc)
        * MW rotation curve from galpy MWPotential2014
        * Dispersions in (Vr,Vphi,Vz) from Tarricq et al. (2021)
    """
    # Initialise the 3D positions of particles:
    R = np.sqrt( x**2 + y**2 )
    rho_astropy = R * u.kpc # distance to galactic centre
    phi_astropy = ( 90 - np.degrees(( np.arctan2(x, y)  ))  )  * u.degree
    z_astropy = z * u.kpc
    particle_rep = coord.CylindricalRepresentation(
        rho=rho_astropy,
        phi=phi_astropy,
        z=z_astropy,
    )
    # Initialise their velocities:
    #circ_velocity = 220 * u.km / u.s # uses 220 for everyone
    #circ_velocity = galpy.potential.vcirc(MWPotential2014,R=rho_astropy,ro=8.,vo=220.) * u.km / u.s
    circ_velocity = np.polyval( [ 2.15296517e-05, -3.14712607e-03,  1.75497780e-01, -5.08984260e+00,
        2.50851095e+02] , R ) 
    # (polynomial approximation valid from 7 to 50 kpc)
    
    age_Gyr = 10**(logt-9)
    sigma_vR = 20.2 * age_Gyr**0.25 * factor
    all_vR = np.random.normal(loc=0, scale=sigma_vR, size=len(x))
    #
    sigma_vPhi = 13.9 * age_Gyr**0.23 * factor
    all_vPhi = (circ_velocity + np.random.normal(loc=0, scale=sigma_vPhi, size=len(x))) * u.km / u.s
    #
    sigma_vZ = 7.9 * age_Gyr**0.19 * factor
    all_vZ = np.random.normal(loc=0, scale=sigma_vZ, size=len(x))
    
    angular_velocity = (-all_vPhi / rho_astropy).to(
        u.mas / u.yr, u.dimensionless_angles()
    )
    particle_dif = coord.CylindricalDifferential(
        d_rho= all_vR * u.km / u.s,
        d_phi= angular_velocity,
        d_z=   all_vZ * u.km / u.s,
    )    
    # Merge and transform:
    particle_rep = particle_rep.with_differentials(particle_dif)
    ooo_gal = coord.SkyCoord(particle_rep, frame=coord.Galactocentric, 
                             z_sun=21 * u.pc, 
                             galcen_distance=8. * u.kpc)
    ooo_obs = ooo_gal.transform_to(coord.ICRS) 
    return ooo_obs.pm_ra_cosdec.value, ooo_obs.pm_dec.value, all_vR, all_vPhi, all_vZ