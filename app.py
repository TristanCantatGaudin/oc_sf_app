import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def mock_mw(N=2000,scaleLength=3., scaleHeight8=0.2, flaring=-0.1, minR=8, maxR=30):
    """ 
    Mock Milky Way
    
    Input:
        N: number of desire points
        scaleLength: scale length, in kpc
        scaleHeight8: scale height at R=8kpc, in kpc
        flaring: inverse flaring scale in kpc 
                (0=no flare, -0.1=height increases by 2.7 every 10kpc)
        minR: do not return points with R smaller than minR in kpc
        
    Output:
        X,Y,Z
    """
    X_to_return = []
    Y_to_return = []
    Z_to_return = []
    

    while len(X_to_return)<N:
        X = np.random.uniform(low=-maxR,high=maxR,size=10000)
        Y = np.random.uniform(low=-maxR,high=maxR,size=10000)
        Z = np.random.uniform(low=-3,high=3,size=10000)
        R = np.sqrt(X**2 + Y**2)

        scaleHeight = scaleHeight8 / np.exp( flaring*(R-8) )
        radial_density = np.exp(-R/scaleLength)
        vertical_density = np.exp(-np.abs(Z)/scaleHeight)/scaleHeight

        ticket = np.random.uniform(low=0,high=np.exp(-minR/scaleLength),size=10000)
        score = radial_density*vertical_density
        keep = (ticket<score) & (R>minR)
        
        X_to_return = X_to_return+X[keep].tolist()
        Y_to_return = Y_to_return+Y[keep].tolist()
        Z_to_return = Z_to_return+Z[keep].tolist()

    X_to_return = np.array(X_to_return[:N])
    Y_to_return = np.array(Y_to_return[:N])
    Z_to_return = np.array(Z_to_return[:N])
    
    return X_to_return,Y_to_return,Z_to_return
    
    

    
col1, col2 = st.columns(2)
with col1:
	st.title('Milky Way')
	scaleLength  = st.slider("scale length (kpc)", min_value=1., max_value=5., value=3.,step=0.01)
	scaleHeight8 = st.slider("scale height (kpc)", min_value=0.1, max_value=1., value=0.2,step=0.01)
	flaring = st.slider("flaring (/kpc)", min_value=-0.2, max_value=0.2, value=-0.1,step=0.01,
	                    help="negative means higher scale height in outer disc, following equation (7) in Bovy et al. (2016ApJ...823...30B)")
	mark = st.checkbox('Mark undetectable clusters',value=True)
    
    	
with col2:
	st.title('Clusters')
	cl_age  = st.slider("age (log t)", min_value=6.5, max_value=9.5, value=9.,step=0.01)
	cl_mass = st.slider("mass (Msun)", min_value=350, max_value=5000, value=355,step=1)
	N  = st.slider("number of clusters", min_value=1000, max_value=50000, value=40000,step=100,
	                help="beyond R=8kpc only")


X,Y,Z = mock_mw(N=int(N),
                scaleLength=float(scaleLength),
			 	scaleHeight8=float(scaleHeight8), 
			 	flaring=float(flaring))
R = np.sqrt(X**2 + Y**2)

pmra  = 0*np.ones(int(N))
pmdec = 0*np.ones(int(N))
age = cl_age*np.ones(int(N))
mass = cl_mass*np.ones(int(N))

# To Galactic coordinates
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
# convert to ICRS
all_stars = SkyCoord(x=X*u.kpc, y=Y*u.kpc, z=Z*u.kpc,
               frame=coord.Galactocentric(galcen_distance=8*u.kpc,z_sun=0*u.kpc), representation_type='cartesian', differential_type='cartesian')
all_stars_Galactic = all_stars.transform_to(coord.Galactic)

is_in_box = all_stars_Galactic.l.value > 140
is_in_box[ all_stars_Galactic.l.value>220] = False
is_in_box[ all_stars_Galactic.b.value<-10] = False
is_in_box[ all_stars_Galactic.b.value>10] = False


with open('run1_XGBOOST_model.pkl', 'rb') as f:
    bst = pickle.load(f)
    
all_features = [all_stars_Galactic.l.value,
                all_stars_Galactic.b.value,
                pmra,pmdec,
                1000*all_stars_Galactic.distance.value,
                age, mass]
all_features = np.array(all_features).T

predicted_CST = bst.predict(all_features)
    



fig_XY_RZ = plt.figure()
#
plt.subplot(121,aspect=1)
plt.scatter( X , Y , s=1 , c='#CCCCCC', zorder=1)
plt.scatter( X[is_in_box] , Y[is_in_box] , s=1 ,zorder=2)
if mark:
    plt.scatter( X[(is_in_box)&(predicted_CST<4)] , Y[(is_in_box)&(predicted_CST<4)] ,
                 s=1 ,zorder=2, c='red')
plt.xlabel('X (kpc)'); plt.ylabel('Y (kpc)')
plt.xlim(-30,30); plt.ylim(-30,30)
plt.minorticks_on()
plt.plot([-8],[0],'o',c='hotpink')
#
plt.subplot(122,aspect=5)
plt.scatter( R[is_in_box] , Z[is_in_box] , s=1 )
if mark:
    plt.scatter( R[(is_in_box)&(predicted_CST<4)] , Z[(is_in_box)&(predicted_CST<4)] ,
                 s=1 ,zorder=2, c='red')
plt.xlabel('R (kpc)'); plt.ylabel('Z (kpc)')
plt.xlim(8,30); plt.ylim(-2.2,2.2)
plt.minorticks_on()
#
plt.tight_layout()

st.pyplot(fig_XY_RZ)




figLB = plt.figure(figsize=(8,3))
plt.scatter( all_stars_Galactic.l.value, all_stars_Galactic.b.value, 
            c=all_stars_Galactic.distance.value, s=5,
			vmin=2, vmax=12 )
plt.colorbar(label='distance (kpc)',extend='both')
if mark:
    plt.scatter( all_stars_Galactic.l.value[(is_in_box)&(predicted_CST<4)] ,
                 all_stars_Galactic.b.value[(is_in_box)&(predicted_CST<4)] ,
                 s=5 ,zorder=2, c='red')
plt.xlim(220,140)
plt.ylim(-10,10)
plt.xlabel('$\ell$ (degrees)'); plt.ylabel('$b$ (degrees)')
st.pyplot(figLB)

