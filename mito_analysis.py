import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# needed if runnign as standaline 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu,pearsonr,spearmanr
from statsmodels.tsa.stattools import ccf
import trackpy as tp
from skimage.measure import label, regionprops
from skimage.morphology import disk,remove_small_objects
import glob
import skimage.io as io
import cv2
import skimage

def mito_tracking(
                pixel_size=11,
                minmass=50,
                maxDisp=50,
                memory=2,
                interval=5,
                nSeg=10,
                force_thresh=800,
                ar_mito=None,
                ar_f=None,
                 ):
    
    # compute trajectories by intervals
    trajs=[]
    for seg in range(nSeg):
        ar_tmp=ar_mito[seg*interval:(seg+1)*interval,:,:].copy()
        minFrame=interval
        f = tp.batch(ar_tmp, pixel_size, minmass=minmass, characterize=True)
        t_before = tp.link(f, maxDisp, memory=memory)
        traj = tp.filter_stubs(t_before, minFrame)
        trajs.append(traj)

    # separate trajs into ones that overlap with forces and ones that don't 
    trajs_force_overlap=[]
    trajs_force_nonoverlap=[]
    for seg in range(nSeg):
        traj=trajs[seg]
        force_mask=ar_f[seg,:,:].copy()
        force_mask[force_mask<force_thresh]=0
        force_mask[force_mask>=force_thresh]=1

        df1=pd.DataFrame(columns=traj.columns)
        df2=pd.DataFrame(columns=traj.columns)
        for j in traj['particle'].unique():
            xs=traj[traj['particle']==j]['x'].tolist()
            ys=traj[traj['particle']==j]['y'].tolist()
            sub_df=traj[traj['particle']==j]
            if force_mask[int(ys[0]),int(xs[0])]==1:
                df1=pd.concat([df1, sub_df])
            else:
                df2=pd.concat([df2, sub_df])

        trajs_force_overlap.append(df1)    
        trajs_force_nonoverlap.append(df2)   
        
    trajectories={
    'all':trajs,
    'overlap':trajs_force_overlap,
    'nonoverlap':trajs_force_nonoverlap
    }
      
    return trajectories

def get_forcear(
                ar_f=None,
                nSeg=10,
                interval=5,
                thresh=800,
                mapcolor='GnBu',
                 ):
    # compute traction development by intervals
    ar_f_seg=np.zeros((nSeg,ar_f.shape[1],ar_f.shape[2]))

    for seg in range(nSeg):
        ar_tmp=ar_f[seg*interval:(seg+1)*interval,:,:].copy()
        ar_tmp=np.mean(ar_tmp,axis=0)
        ar_f_seg[seg]=ar_tmp

    gradient=np.linspace(0.1,1,nSeg)
    ar_f_stacked=np.zeros((ar_f.shape[1],ar_f.shape[2]))

    for seg in range(nSeg):
        ar_tmp=ar_f_seg[seg,:,:].copy()
        if np.max(ar_tmp) > 5000: # correct for out-of-focus frames
            ar_tmp=ar_f_seg[seg-1,:,:].copy()
        ar_tmp[ar_tmp<thresh]=0
        ar_tmp[ar_tmp>=thresh]=1 
        ar_f_stacked[ar_tmp==1]=gradient[seg]

    fig=plt.figure(figsize=(5,5))
    plt.imshow(ar_f_stacked,cmap=mapcolor) 

    return ar_f_seg,ar_f_stacked

def plot_traj(
            nSeg=5,
            ar_f=None,
            trajs=None,
            trajcolor='Purples',
            forcecolor='GnBu',
            ):
    
    cmap=matplotlib.colormaps[trajcolor]
    gradient=np.linspace(0.1,1,nSeg)
    pos=plt.imshow(ar_f,cmap=forcecolor) 
    for seg in range(nSeg):
        traj=trajs[seg]
        for j in traj['particle'].unique():
            xs=traj[traj['particle']==j]['x'].tolist()
            ys=traj[traj['particle']==j]['y'].tolist()
            nFrames=len(xs)
            cs=np.linspace(0,1,nFrames)
            plt.plot(xs,ys,c=cmap(gradient[seg]),lw=0.6)
            plt.scatter(xs[-1],ys[-1],color=cmap(gradient[seg]),s=1)

            
def plot_distribution(
                    data,
                    cols=None,
                    alpha=0.5
                    ):

    parts=plt.violinplot(data,showextrema=False)
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor(cols[i])
        pc.set_alpha(0.3)
    plt.boxplot(data,showfliers=False)
    for i,vals in enumerate(data):
        x=np.random.normal(i+1,0.05,len(vals))
        plt.scatter(x,vals,color=cols[i],alpha=alpha,s=10)
        
        
def plot_timeseries(
                    data,
                    color='mediumpurple',
                    ):

    parts=plt.violinplot(data,showextrema=False)
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    plt.boxplot(data,showfliers=False)
    for i,vals in enumerate(data):
        x=np.random.normal(i+1,0.05,len(vals))
        plt.scatter(x,vals,color=color,alpha=0.3,s=10)

        
def extract_mitodynamics(frameRate=20,
                 interval=5,
                 distThresh_lower=100,
                 distThresh_upper=400,
                 treat=90,
                 nSeg=10,
                 ar_mito=None,
                 ar_cyto=None,
                 ar_f=None,
                 center=[600,600],
                 trajs=None,
                 path=None,
                ):
    
    # add in flow direction information
    trajs_new=[]

    for seg in range(nSeg):
        traj=trajs[seg]
        
        # for 'overlap' and 'nonoverlap' trajs, there could be frames without particles; we'll ignore those frames
        try: 
            dpr = traj.groupby('particle').apply(lambda g: get_dotproduct(g, center)).rename('dot_product')
        except: 
            IndexError
        direction = dpr.apply(classify_by_dpr).rename('direction')

        traj_new = traj.merge(dpr, on='particle')
        traj_new = traj_new.merge(direction, on='particle')

        trajs_new.append(traj_new)
    
    
    traction_file=path+'/TFM_analysis.csv'
    df=pd.read_csv(traction_file)
    es=df['energy_per_area'].values
    ts=np.linspace(0,len(es)*frameRate/60,len(es))
    
    locs=['nuc','peri']
    dirs=['retrograde','anterograde']
    
    # create dict to store data
    data={}
    for loc in locs:
        data[loc]={}
        for direction in dirs:
            data[loc][direction]={  
                    'vs':[],
                    'vs_temporal':{'before':[],'after':[]},
                    'mitoCa_temporal':{'before':[],'after':[]},
                    'cytoCa_temporal':{'before':[],'after':[]},
                    'vs_time':[],
                    'mitoCa_time':[],
                    'cytoCa_time':[],
                    'esLocal_time':[],
                    'size_time':[],
                    'ecc_time':[],
                    'es_time':[],
                    'count_time':[],                
                    }

    # collect data
    for seg in range(nSeg):
        start=int(seg*interval)
        end=int((seg+1)*interval)
        traj=trajs_new[seg]

        sub_data={}
        for loc in locs:
            sub_data[loc]={}
            for direction in dirs:
                sub_data[loc][direction]={'vs_time':[],'mitoCa_time':[],'cytoCa_time':[],'esLocal_time':[],
                                         'size_time':[],'ecc_time':[],'count_time':0}

        for j in traj['particle'].unique():
            tj=traj[traj['particle']==j]

            # get size and eccentricity (0: circle, 1: elongated) info
            size=np.mean(tj['size'].values)
            ecc=np.mean(tj['ecc'].values)

            # get coordinates and distance to nucleus
            xs=tj['x'].tolist()
            ys=tj['y'].tolist()
            nucDist=np.sqrt((xs[0]-center[1])**2+(ys[0]-center[0])**2)
            dist=np.sqrt((xs[0]-xs[-1])**2+(ys[0]-ys[-1])**2)*11 # 1 pixel==11 um
            vel=dist/interval/frameRate
            # get Ca info
            xs_tmp=[int(x) for x in xs]
            ys_tmp=[int(y) for y in ys]
            ts=np.arange(start,end)
            caVals_mito=[]
            caVals_cyto=[]
            esVals=[]
            mask_time=np.zeros_like(ar_mito)

            for i,t in enumerate(ts):
                x=xs_tmp[i]
                y=ys_tmp[i]
                coords=skimage.draw.disk((x,y),10,shape=ar_mito[0,:,:].shape) # radius=10 pixels
                mask=np.zeros_like(ar_mito[0,:,:])
                mask[coords]=1
                mask_time[i]=mask
                ar_mito_tmp=ar_mito.copy()*mask
                ar_cyto_tmp=ar_cyto.copy()*mask
                val_mito=np.mean(ar_mito_tmp[t,:,:])
                val_cyto=np.mean(ar_cyto_tmp[t,:,:])
                caVals_mito.append(val_mito)
                caVals_cyto.append(val_cyto)

            tfm_df=get_force(path=path,forcemask=mask_time)
            esVals=tfm_df['energy_pJ']

            ca_mito=np.mean(caVals_mito)
            ca_cyto=np.mean(caVals_cyto)
            es_local=np.mean(esVals)

            # nuclear
            if nucDist<distThresh_lower:
                loc='nuc'
            else:
                loc='peri'    
            direction=tj['direction'].values[0]

            # velocity
            if seg >= int(treat/interval)-5 and seg < int(treat/interval):
                data[loc][direction]['vs_temporal']['before'].append(vel)
            elif seg >= int(treat/interval) and seg < int(treat/interval)+5:
                data[loc][direction]['vs_temporal']['after'].append(vel)
            # Ca
            if seg >= int(treat/interval)-2 and seg < int(treat/interval):
                data[loc][direction]['mitoCa_temporal']['before'].append(ca_mito)
                data[loc][direction]['cytoCa_temporal']['before'].append(ca_cyto)
            elif seg >= int(treat/interval) and seg < int(treat/interval)+2:
                data[loc][direction]['mitoCa_temporal']['after'].append(ca_mito)
                data[loc][direction]['cytoCa_temporal']['after'].append(ca_cyto)

            # time-series
            data[loc][direction]['vs'].append(vel)
            sub_data[loc][direction]['vs_time'].append(vel)
            sub_data[loc][direction]['mitoCa_time'].append(ca_mito)
            sub_data[loc][direction]['cytoCa_time'].append(ca_cyto)
            sub_data[loc][direction]['esLocal_time'].append(es_local)
            sub_data[loc][direction]['size_time'].append(size)
            sub_data[loc][direction]['ecc_time'].append(ecc)
            sub_data[loc][direction]['count_time']+=1

        e=es[start:end]

        for loc in locs:
            for direction in dirs:
                for key in sub_data[loc][direction].keys():    
                    data[loc][direction][key].append(sub_data[loc][direction][key])
                data[loc][direction]['es_time'].append(e)

    return data
        
        
def bootstrap_diff(data1,data2,nIter=1000):
    diffs=[]
    nSample=int(0.3*len(data1))
    for i in range(nIter):
        # set1
        data=np.array(data1)
        nData=len(data)
        idxs=np.random.choice(nData,nSample)
        set1=data[idxs]
        # set2
        data=np.array(data2)
        nData=len(data)        
        idxs=np.random.choice(nData,nSample)
        set2=data[idxs]
        # difference between set2 and set1
        diff=set2-set1
        mean=np.mean(diff)
        diffs.append(mean)
        
    fig=plt.figure(figsize=(3,2))
    ax=fig.add_subplot(111)
    ax.set_xlabel('difference')
    ax.set_ylabel('PDF')
    sns.kdeplot(diffs)
    kde_lines = ax.get_lines()[-1]
    kde_x, kde_y = kde_lines.get_data()
    thresh=0
    mask = kde_x < thresh
    area=np.sum(kde_y[mask])
    print('Area below 0: {}'.format(area))
    mask = kde_x > thresh
    area=np.sum(kde_y[mask])
    print('Area above 0: {}'.format(area))
    filled_x, filled_y = kde_x[mask], kde_y[mask]
    ax.fill_between(filled_x, y1=filled_y,alpha=0.5)        
    ax.axvline(thresh,linestyle='--',alpha=0.5)

    return diffs


def mwTest(data1,data2,method='less'):
    u_statistic, p_value = mannwhitneyu(data1, data2, alternative='less')
    return p_value


def permutation_test(data1,data2,nIter=10000):
    data1=np.array(vels2['before'])
    data2=np.array(vels2['after'])
    # Combine the groups into a single dataset
    combined = np.append(data1,data2)
    obs_diff = np.mean(data1)-np.mean(data2)

    perm_diffs = []
    for i in range(nIter):
        permuted = np.random.permutation(combined)
        perm_group1 = permuted[:len(data1)]
        perm_group2 = permuted[len(data1):]
        diff = np.mean(perm_group2) - np.mean(perm_group1)
        perm_diffs.append(diff)

    # Calculate the p-value as the proportion of permuted differences that are greater than or equal to the observed difference
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return p_value


def correlation(data1,data2,crosscorr=False):
    correlation_pearson, p_value_pearson = pearsonr(data1, data2)
    print('Pearson correlation R: {}, P value: {}'.format(correlation_pearson, p_value_pearson))

    correlation_spearman, p_value_spearman = spearmanr(data1, data2)
    print('Spearman correlation R: {}, P value: {}'.format(correlation_spearman, p_value_spearman))
    
    # Calculate cross-correlation function
    if crosscorr:
        ccf_values = ccf(data1, data2, adjusted=True)
        fig=plt.figure(figsize=(5,3))
        ax=fig.add_subplot(111)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross-correlation')
        plt.plot(ccf_values,color='dodgerblue')        

    return correlation_pearson, p_value_pearson, correlation_spearman, p_value_spearman
        
def mito_density(
                ar_mito=None,
                ar_cell=None,
                ar_f=None,
                force_thresh=800,
                mito_thresh=105,
                center=[600,600],
                path=None,
                small_object_size=500,
                dist_thresh=300,
                frameRate=20,
                ):
    
    fig=plt.figure(figsize=(20,10))
    ax=fig.add_subplot(131)
    
    forcemask=ar_f.copy()
    forcemask[forcemask<force_thresh]=0
    forcemask[forcemask>=force_thresh]=1
    forcemask = label(forcemask)
    ax.imshow(forcemask)
    ax.scatter(center[1],center[0],color='white') 
    
    props = regionprops(forcemask)
    for area,prop in enumerate(props):
        c=prop.centroid
        dist=np.sqrt((c[0]-center[0])**2+(c[1]-center[1])**2)
        if dist>dist_thresh:
            forcemask[forcemask==area+1]=0
   
    forcemask = remove_small_objects(forcemask, small_object_size)
    dilation_size = 10
    forceSE = disk(dilation_size)
    forcemask = cv2.dilate(forcemask.astype('uint8'), forceSE)

    # relabel
    forcemask = label(forcemask)
    nMask=np.max(forcemask)
    ax=fig.add_subplot(132)
    ax.imshow(forcemask)
    ax.scatter(center[1],center[0],color='white') 
    for area in range(1,nMask+1):
        idx=np.argwhere(forcemask==area)[0]
        ax.text(idx[1], idx[0], int(area),ha="center", va="center", fontsize=20, color='orange')
    
    ar_mito_threshed=ar_mito.copy()
    ar_mito_threshed[ar_mito_threshed<mito_thresh]=0
    ar_mito_threshed[ar_mito_threshed>=mito_thresh]=1
    
    ax=fig.add_subplot(133)
    ax.imshow(ar_mito_threshed[0,:,:])
    
    local_mitoCa=[]
    local_cytoCa=[]
    local_mito_density=[]
    local_force=[]
    fig=plt.figure(figsize=(10,12))
    for i in range(1,nMask+1):
        ax=fig.add_subplot(6,3,i)
        ax.set_title('area {}'.format(i))
        ax.set_xlabel('time (min)')
        mask=forcemask.copy()
        mask[mask!=i]=0
        mask[mask==i]=1
        maskArea=np.sum(mask)
        # mitoCa
        ar_mito_masked=ar_mito.copy()*mask
        mitoCa=np.sum(ar_mito_masked[:,:,:],axis=(1,2))/maskArea
        local_mitoCa.append(mitoCa)
        # cytoCa
        ar_cyto_masked=ar_cell.copy()*mask
        cytoCa=np.sum(ar_cyto_masked[:,:,:],axis=(1,2))/maskArea
        local_cytoCa.append(cytoCa)
        # mito density
        ar_mito_masked=ar_mito_threshed.copy()*mask
        dens=np.sum(ar_mito_masked[:,:,:],axis=(1,2))
        local_mito_density.append(dens)
        # force
        maskar=np.array([mask]*ar_mito.shape[0])
        tfm_df=get_force(path=path,forcemask=maskar)
        es=tfm_df['energy_pJ']
        local_force.append(es)
        
        ts=np.linspace(0,frameRate*len(dens)/60,len(dens))
        
        ax.plot(ts,dens,color='dodgerblue',alpha=0.8)
        axr=ax.twinx()
        axr.plot(ts,es,color='mediumpurple',alpha=0.8)

    plt.tight_layout()
        
    return forcemask,local_mito_density,local_force, local_mitoCa, local_cytoCa


# Adapt from TFM_analysis(GFP='', force_min=0)"
# Read in the CSV file with all the file details and convert to a dictionary
def get_force(
            path=None,
            forcemask=None, # binary 
            force_min=100,
             ):
    
    temp_dict = pd.read_csv(path+'/TFM_params.csv')
    TFM_params = {}
    for key in temp_dict.keys()[1:]:
        TFM_params[key] = temp_dict[key][0]

    # Find the displacement and traction stress files
    file_list_fx = sorted(glob.glob(path+'/traction_files/fx_*.tif'))
    file_list_fy = sorted(glob.glob(path+'/traction_files/fy_*.tif'))
    file_list_dispu = sorted(glob.glob(path+'/displacement_files/disp_u*.tif'))
    file_list_dispv = sorted(glob.glob(path+'/displacement_files/disp_v*.tif'))

    # Number of files to process
    N_images = len(file_list_fx)

    forcemask = forcemask.astype(bool)

    # create empty variables to store all the data
    energy = []
    #energy_per_area = []
    #residual = []
    force_sum = []
    displacement_sum = []
    time = []
    #cell_area = []

    # loop over all the frames in the series
    for timepoint in np.arange(0,N_images):
        # correct for missing frames. When there are missing frames, set value to that from the previous frame with values 
        #while np.unique(forcemask[timepoint].ravel()).shape[0] != 2:
        #    timepoint-=1

        mask_tmp=forcemask[timepoint]

        # read in the file
        tractionx = io.imread(file_list_fx[timepoint])
        tractiony = io.imread(file_list_fy[timepoint])
        dispx = io.imread(file_list_dispu[timepoint])
        dispy = io.imread(file_list_dispv[timepoint])

        # only use points in the forcemask
        tractionx = tractionx[mask_tmp]
        tractiony = tractiony[mask_tmp]
        dispx = dispx[mask_tmp]
        dispy = dispy[mask_tmp]

        if force_min > 0:
            # calculate the magnitude at each pixel
            traction_mag = np.sqrt(tractionx**2 + tractiony**2)
            traction_thresh_mask = traction_mag > force_min
            tractionx = tractionx[traction_thresh_mask]
            tractiony = tractiony[traction_thresh_mask]
            dispx = dispx[traction_thresh_mask]
            dispy = dispy[traction_thresh_mask]

        # energy is one half the sum of the dot product of the traction vector with the displacement vector
        # need to include corrections for the units and the area covered. 10^-6 is to put the number in pJ
        energy.append( 0.5 * np.sum(((dispx * tractionx) + dispy * tractiony)) * TFM_params['mesh_size']**2 * TFM_params['um_per_pixel']**3 * 10**-6)

        # force_sum is the sum of the absolute magnitudes of the vectors in the mask
        force_sum.append( np.sum( np.sqrt(tractionx**2 + tractiony**2)))

        # sum of the displacement magnitudes
        displacement_sum.append( np.sum( np.sqrt(dispx**2 + dispy**2)))

        # residual is an error metric. Should be less than 0.1 (e.g. 10%)
        #residual.append( np.sqrt( np.sum(tractionx) ** 2 + np.sum(tractiony) ** 2) / force_sum[timepoint] * 100)

        # stores the time point
        time.append(timepoint)
        
        # calculate cell area
        #cell_area.append(np.sum(forcemask) * (TFM_params['um_per_pixel'] ** 2))
        
        # calculate energy per area
        #energy_per_area.append(energy[timepoint] / cell_area[timepoint])
    
    
    # Convert the lists of data to a dictionary and save it as a CSV file
    TFM_analysis_dict = {
        'time': time,
        #'cell_area_microns2' : cell_area,
        'force_minimum' : [force_min] * N_images,
        'force_sum_Pa': force_sum,
        'displacement_sum': displacement_sum,
        #'residual': residual,
        'energy_pJ': energy,
        #'energy_per_area': energy_per_area,
        }
    TFM_dataframe = pd.DataFrame(TFM_analysis_dict)

    return TFM_dataframe
        
        
def get_dotproduct(df,center):
    start = df.iloc[0][['x', 'y']].values
    end = df.iloc[-1][['x', 'y']].values
    vec = end - start
    ref = end - np.array(center)
    dpr = np.dot(vec, ref)

    return dpr

def classify_by_dpr(dpr):
    return 'anterograde' if dpr>0 else 'retrograde'       
        
        