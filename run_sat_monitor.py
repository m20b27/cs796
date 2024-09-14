# %% [markdown]
# ### Imports and helper functions
# import libraries

# %%
from skyfield.api import load, wgs84
import math
import numpy as np
import logging
import time
import datetime
from datetime import datetime,timezone, timedelta
import starlink_grpc

import matplotlib.pyplot as plt
import copy

# %% [markdown]
# gRPC helper methods

# %%

def GRPC_ResetMap(context):
    try:
        starlink_grpc.reset_obstruction_map(context)
    except:
        print("Unable to reset obstruction map")

def GRPC_GetDishyLocation(context):
    try:
        location_data = starlink_grpc.location_data(context)
    except:
        print("Unable to reset obstruction map")
    
    dishy_latitude = location_data["latitude"]
    dishy_longitude = location_data["longitude"]
    dishy_altitude = location_data["altitude"]
    print(f"dishy lat: {dishy_latitude}, long: {dishy_longitude}, alt: {dishy_altitude}")
    # return wgs84.latlon(dishy_latitude, dishy_longitude, dishy_altitude) 
    return wgs84.latlon(dishy_latitude, dishy_longitude, elevation_m=26) 

def GRPC_GetObstructionMap(context):
    try:
        snr_data = starlink_grpc.obstruction_map(context)
        return snr_data
    except starlink_grpc.GrpcError as e:
        logging.error("Failed getting obstruction map data: %s", str(e))
        return 1
    

def GetLatestTLE():
    stations_url = 'http://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle'
    return load.tle_file(stations_url, reload=False)

# %% [markdown]
# other helper methods

# %%
# convert snr tuple to a 2d array
def to2DArray(snr_tuple):
    twoD_arr = np.array(snr_tuple)
    return twoD_arr

# convert the raw snr data into a 2d boolean matrix, set data point with value > 0 to 1, <= 0 to 0
def ConvertToBooleanMatrix(snr):
    # convert to 2d array
    matrix_2d = to2DArray(snr)
    # find indices of elements which < 0
    lessThanZero = (matrix_2d < 0)  # this is a boolean array 
    # mask elements < 0 to 0
    matrix_2d[lessThanZero] = 0
    # find indices of elements which > 0
    greaterThanZero = (matrix_2d > 0)
    # mask elements < 0 to 1
    matrix_2d[greaterThanZero] = 1
    
    return matrix_2d # return a boolean matrix

# apply XOR to two boolean matrices
def GetXorDiff(boolMap1, boolMap2):
    return np.logical_xor(boolMap1, boolMap2)
    
# get satellite elevation and azimuth from the observor's location, well-known library skyfield is used: https://rhodesmill.org/skyfield/earth-satellites.html
def getSatelliteAltAz(timePoint, satellite, ObservorLocation):
    difference = satellite - ObservorLocation
    topocentric = difference.at(timePoint)
    alt, az, distance = topocentric.altaz()
    return (alt.degrees, az.degrees)

# convert elevation to the distance from the centre point of the obstruction map to a point reprensent a satellite's trace point
# using 45 pixels as the radius of the contained polar plot as per Tanveer, Puchol, etc suggested

# linear projection considering the 25 degrees AOE
def elevationToCentreDistance_linearProjection_with_25degrees(el): # el in degrees
    return 45 * (1- ((el-25)/65))

# linear projection not considering the 25 degrees AOE
def elevationToCentreDistance_linearProjection_without_25degrees(el): # el in degrees
    return 45 * (1- (el/90))

# cosine projection considering the 25 degrees AOE
def elevationToCentreDistance_cosineProjection_with_25degrees(el): # el in degrees
    el_minus_25_in_rad =  (el-25) * math.pi/180
    return 45 *(math.cos(el_minus_25_in_rad))


    

# cosine projection not considering the 25 degrees AOE
def elevationToCentreDistance_cosineProjection_without_25degrees(el): # el in degrees
    el_in_rad =  el * math.pi/180
    return  45 * math.cos(el_in_rad)


# using the azimuth and the distance from the centre point to calculate the xp, yp coordinates on the polar plot
def azimuth_elevation_to_XY_projection(az, r): # az in degrees
    az_in_rad = az * math.pi/180
    
    yp = math.sin(math.pi/2 - az_in_rad) * r
    xp = math.cos(math.pi/2 - az_in_rad) * r
    
    return xp, yp
    
# convert xp,yp projections back to the points on an obstruction map matrix
def xpypToBooleanMatrix(xpyps, map_height, map_width):
    matrix = np.empty(shape=(map_height, map_width))
    matrix.fill(False)
    for xpyp in xpyps:
        # convert xp yp to column and row indices
        column = round(xpyp[0] + (map_width/2)) 
        row = round((xpyp[1]* -1) + (map_height/2)) # row from top to bottom
        
        if column >= map_width or column < 0 or row >= map_height or row < 0:
            continue
        
        matrix[row, column] = True
        
    return matrix
    

# %% [markdown]
# ### Main logic
# 1. retrieve the latest TLE data

# %%
allSatsFromTLE = GetLatestTLE()

# %% [markdown]
# 2. retrieve the dishy location and print out the dishy status

# %%
context = starlink_grpc.ChannelContext()
dishyLocation = GRPC_GetDishyLocation(context)

print(starlink_grpc.get_status(context))

# %% [markdown]
#  
# 4. i. reset dishy's obstruction map   
#    ii. record the current utc time as t0, then retrieve the obstruction map as obstruction0   
#    iii. then wait for a certain time period for the dishy's obstruction map to be filled out   
#    iv. finally record the current utc time again as t1, and retrieve the obstruction map as obstruction1

# %%
GRPC_ResetMap(context)

LOOP_TIME_DEFAULT = 15

t0 = datetime.now(timezone.utc)
print("t0: ", t0)
obstruction0 = GRPC_GetObstructionMap(context)

time.sleep(LOOP_TIME_DEFAULT)

t1 = datetime.now(timezone.utc)
obstruction1 = GRPC_GetObstructionMap(context)



# %% [markdown]
# 5. i. convert obstruction0 and obstruction1 into boolean matrices   
#   ii. flip obstruction0 and obstruction1 horizontally as the obstruction map is mirrored horizontally to the actual direction    
#   iii. mask all the element with value of False to np.nan so these elements will be transparent once painted with plt package   
#   iv. display obstruction0 and obstruction1   
#    

# %%
red_cmap = plt.cm.colors.ListedColormap(['red'])
red_cmap.set_bad(alpha=0)

bool0 = ConvertToBooleanMatrix(obstruction0)
bool0 = np.flip(bool0, 1) # horizontal flip
bool0_masked = np.where(bool0 == False, np.nan, bool0)


bool1 = ConvertToBooleanMatrix(obstruction1)
bool1 = np.flip(bool1, 1) # horizontal flip
bool1_masked = np.where(bool1 == False, np.nan, bool1)


# %% [markdown]
# 6. isolated the trajectories with a XOR operation applying to obstruction0 and obstruction1

# %%
xor_result = GetXorDiff(bool0, bool1)

xor_result_masked = np.where(xor_result == False, np.nan, xor_result)
plt.imshow(xor_result_masked, interpolation='none', cmap=red_cmap, alpha=0.5)



# %% [markdown]
# find obstruction map width and height

# %%
map_width = len(obstruction0)
map_height = len(obstruction0[0])
print(f"map width: {map_width}, map height: {map_height}")

# %%
def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    # if theta < 0:
    #      theta += 2 * math.pi
    return r, (math.pi/2) - theta

# from https://stackoverflow.com/questions/20296030/what-algorithm-can-i-use-to-recognize-the-line-in-this-scatterplot
def count_diag_groups(x, y, width):
    d = x - y
    result = []
    for i in range(d.size):
        delta = d - d[i]
        neighbors = np.where((delta >= 0) & (delta <= width))[0]
        result.append(neighbors)
    return result

# from https://stackoverflow.com/questions/20296030/what-algorithm-can-i-use-to-recognize-the-line-in-this-scatterplot
def findBestLine(x, y):
        # Find a selection of points that are close to being aligned
    # with a slope of 1.
    width = 5
    r = count_diag_groups(x, y, width)

    # Find the largest group.
    sz = np.array(list(len(f) for f in r))
    imax = sz.argmax()
    # k holds the indices of the selected points.
    selection = r[imax]
    

    return selection

# get theta and r

trueIndex = np.argwhere(xor_result==True)
print(trueIndex)

rows, colomns = zip(*trueIndex)
rows = np.array(rows)
colomns = np.array(colomns)

xs = colomns
ys = rows

selected = findBestLine(xs, ys)


rThetas = []

for selected_ind in selected:
    yp = 62 - ys[selected_ind]
    xp = xs[selected_ind] - 62
    r, theta = cartesian_to_polar(xp, yp)
    rThetas.append((r, theta))



# %%
ts = load.timescale()

timespan = (t1-t0).total_seconds()
# build array of timepoints
timePeriod = []
for delta in range(int(timespan)):
    t = t0 + timedelta(seconds=delta)
    timePeriod.append(ts.from_datetime(t))

print(f"number of timepoints: {len(timePeriod)}")
print('UTC date and time:', timePeriod[0].utc_strftime())

# %%

tleTraces = [] # array of (r, az)

tleTracesCartisian = [] # array of [xp, yp]
candidateSats = [] # array of sats

def elevationToR(el): #in degrees
    return 45 * (1- ((el-25)/65))

    
def polar_to_cartesian(r, theta):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.array([x, y])


for sat in allSatsFromTLE:
    
    rAzs = [] # in radians
    #vectors = [] #ENU vector
    cartisians = []
    # get candidate satellites (visable by the dishy which means elevation > 25 degrees) trajetories of the time slot
    for timepoint in timePeriod:
        altaz_in_degrees = getSatelliteAltAz(timepoint, sat, dishyLocation)
        if altaz_in_degrees[0] > 25: #if elevation > 25 degrees
            r = elevationToR(altaz_in_degrees[0])
            rAzs.append((r,math.radians(altaz_in_degrees[1])))
            cartisians.append(polar_to_cartesian(r, math.radians(altaz_in_degrees[1])))
            
    
    if len(rAzs) > 0:
        tleTraces.append(rAzs)
        tleTracesCartisian.append(cartisians)
        candidateSats.append(sat)
        
        days = timePeriod[0] - sat.epoch
        print('{}: {:.3f} days away from epoch'.format(sat.name, days))

# %%
from tslearn.metrics import dtw as ts_dtw



measureCartisians = []

for rtheta in rThetas:
     cart = polar_to_cartesian(rtheta[0], rtheta[1])
     measureCartisians.append(cart)


min_distance = math.inf
second_min_distance = math.inf

for i in range(len(tleTracesCartisian)):
    distance = ts_dtw(measureCartisians, tleTracesCartisian[i])
    if distance < min_distance:
        if min_distance < second_min_distance:
            second_min_distance = min_distance
            second_best_sat = best_sat
            second_best_sat_trace =  best_sat_trace
            
        min_distance = distance
        best_sat = candidateSats[i]
        best_sat_trace = tleTraces[i]
        
    
    
    elif distance < second_min_distance:
        second_min_distance = distance
        second_best_sat = candidateSats[i]
        second_best_sat_trace = tleTraces[i]
    
    
    

# %%
NESW = ["N", "E", "S", "W"]
direction_angles = [0, 90, 180, 270]

fig, axs = plt.subplots(1, 2, subplot_kw={'polar': True}, figsize=(12, 7))
axs = np.array([axs])


axs[0,0].set_title('Traces from obstruction map')
axs[0,1].set_title('Traces from TLE with highlighted best matches')


rThetas = np.array(rThetas)

axs[0,0].scatter(rThetas[:,1], rThetas[:,0], label="observed trace")



best_sat_trace = np.array(best_sat_trace)
second_best_sat_trace = np.array(second_best_sat_trace)


for trace in tleTraces:
    trace = np.array(trace)
    axs[0,1].scatter(trace[:,1], trace[:,0], cmap='viridis', alpha=0.05)
    
best_days = timePeriod[0] - best_sat.epoch
print('best sat {}: {:.3f} days away from epoch'.format(best_sat.name, best_days))

second_best_days = timePeriod[0] - second_best_sat.epoch
print('second best sat {}: {:.3f} days away from epoch'.format(second_best_sat.name, second_best_days))

axs[0,1].scatter(best_sat_trace[:,1], best_sat_trace[:,0], c="red", label="best: "+ best_sat.name+ "\nTLE " + str(round(best_days,3)) +" days away from epoch")
axs[0,1].scatter(second_best_sat_trace[:,1], second_best_sat_trace[:,0], c="blue", label="2nd best: "+ second_best_sat.name + "\nTLE " + str(round(second_best_days,3)) +" days away from epoch")



for ax in axs[0]:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 45)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_thetagrids(angles=direction_angles, labels=NESW, fontsize=12, weight="bold")
    ax.set_rorigin(-0.3)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.2))




plt.savefig("results/from_" + str(t0.isoformat(sep='_', timespec='seconds')) + "_to_" + str(t1.isoformat(sep='_', timespec='seconds')) + '.png')
plt.close(fig)




