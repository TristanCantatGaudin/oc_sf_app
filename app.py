import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mwfunctions import xyz_to_pmradec

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
        X = np.random.uniform(low=-maxR,high=0,size=10000)
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
    
    
st.sidebar.title('Milky Way')
scaleLength  = st.sidebar.slider("scale length (kpc)", min_value=1., max_value=5., value=3.,step=0.01)
scaleHeight8 = st.sidebar.slider("scale height (kpc)", min_value=0.1, max_value=1., value=0.2,step=0.01)
flaring = st.sidebar.slider("flaring (/kpc)", min_value=-0.2, max_value=0.2, value=-0.1,step=0.01,
                    help="negative means higher scale height in outer disc, following equation (7) in Bovy et al. (2016ApJ...823...30B)")
N  = st.sidebar.slider("number of clusters", min_value=1000, max_value=500000, value=40000,step=100,
                help="beyond R=8kpc only")
mark = st.sidebar.checkbox('Flag undetectable clusters (CST<4)',value=True)



col1, col2 = st.columns(2)

with col1:
    st.title('Clusters')
    cl_mass = st.slider("mass (Msun)", min_value=50, max_value=5000, value=355,step=1)

with col2:
    cl_age  = st.slider("age (log t)", min_value=6.5, max_value=9.5, value=9.,step=0.01)
    cl_heating = st.slider("heating", min_value=0., max_value=5., value=1.,step=0.01,
                           help='multiplicative factor to the Tarricq et al. (2021) velocity dispersions')

X,Y,Z = mock_mw(N=int(N),
                scaleLength=float(scaleLength),
                scaleHeight8=float(scaleHeight8), 
                flaring=float(flaring))
R = np.sqrt(X**2 + Y**2)


age = cl_age*np.ones(int(N))
mass = cl_mass*np.ones(int(N))

# assign proper motions
pmra, pmdec, vR, vPhi, vZ = xyz_to_pmradec(X,Y,Z,logt=cl_age, factor=cl_heating)

# To Galactic coordinates
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
# convert to ICRS
all_stars = SkyCoord(x=X*u.kpc, y=Y*u.kpc, z=Z*u.kpc,
               frame=coord.Galactocentric(galcen_distance=8*u.kpc,z_sun=0*u.kpc), representation_type='cartesian', differential_type='cartesian')
all_stars_Galactic = all_stars.transform_to(coord.Galactic)

is_in_box = all_stars_Galactic.l.value > 140
is_in_box[ all_stars_Galactic.l.value>240] = False
is_in_box[ all_stars_Galactic.b.value<-10] = False
is_in_box[ all_stars_Galactic.b.value>10] = False


with open('run1B_XGBOOST_model.pkl', 'rb') as f:
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






figLB = plt.figure(figsize=(8,3))
plt.scatter( all_stars_Galactic.l.value, all_stars_Galactic.b.value, 
            c=all_stars_Galactic.distance.value, s=5,
            vmin=2, vmax=12 )
plt.colorbar(label='distance (kpc)',extend='both')
if mark:
    plt.scatter( all_stars_Galactic.l.value[(is_in_box)&(predicted_CST<4)] ,
                 all_stars_Galactic.b.value[(is_in_box)&(predicted_CST<4)] ,
                 s=5 ,zorder=2, c='red')
plt.xlim(240,140)
plt.ylim(-10,10)
plt.xlabel('$\ell$ (degrees)'); plt.ylabel('$b$ (degrees)')
plt.minorticks_on()





fig_RgcZ = plt.figure(figsize=(8,3))
plt.subplot(111,aspect=2.5)
rbins = np.linspace(8,30,23)
zbins = np.linspace(-2,2,9)
H_in,redges,zedges  = np.histogram2d( R[is_in_box] , Z[is_in_box] , bins=[rbins,zbins])
H_out,_,_ = np.histogram2d( R[(is_in_box)&(predicted_CST>4)] , Z[(is_in_box)&(predicted_CST>4)] , bins=[rbins,zbins])
plt.pcolormesh(redges, zedges, (H_out/H_in).T , cmap='inferno')
plt.xlabel('R (kpc)'); plt.ylabel('Z (kpc)')
plt.colorbar(label='fraction detected')
plt.minorticks_on()
plt.tight_layout()


fig_vel = plt.figure()
plt.subplot(221)
plt.hist(  vZ , bins=50 )
plt.xlabel('V z (km/s)'); plt.ylabel('count')

plt.subplot(222)
plt.scatter( pmra[is_in_box] , pmdec[is_in_box] , s=1 , 
            c=all_stars_Galactic.l.value[is_in_box], vmin=140, vmax=240)
plt.colorbar(label='$\ell$ (degrees)')
plt.xlabel('pmra (mas/yr)'); plt.ylabel('pmdec (mas/yr)')
plt.xlim(-5,5); plt.ylim(-6,4)

plt.subplot(212)
plt.scatter(  R , vPhi , s=1)
plt.xlabel('R (kpc)'); plt.ylabel('V phi (km/s)')
plt.tight_layout()

tab1, tab2, tab3 = st.tabs(["Overview", "Fraction", "Velocities"])

with tab1:
    st.pyplot(fig_XY_RZ)
    st.pyplot(figLB)
with tab2:
    st.pyplot(fig_RgcZ)
with tab3:
    st.pyplot(fig_vel)

# Download as CSV:
output_df = pd.DataFrame(
    {'x': X,
     'y': Y,
     'z': Z,
     'l': all_stars_Galactic.l.value,
     'b': all_stars_Galactic.b.value,
     'd': all_stars_Galactic.distance.value,
     'pmra': pmra,
     'pmdec': pmdec,
     'vR':vR,
     'vPhi':vPhi,
     'vZ':vZ,
     'age': age,
     'mass': mass,
     'cst': predicted_CST,
    })

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(output_df[is_in_box])

st.download_button(
   "Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

