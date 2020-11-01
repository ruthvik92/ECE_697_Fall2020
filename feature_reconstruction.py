import numpy as np
import matplotlib.pyplot as plt

def DoG(window_size, sigma1, sigma2):
    x=np.arange(1,int(window_size)+1,dtype=np.float64)
    x=np.tile(x,(int(window_size),1))
    y=np.transpose(x)
    d2 = ((x-(window_size/2))-.5)**2 + ((y-(window_size/2))-.5)**2
    gfilter = 1/np.sqrt(2*np.pi) * ( 1/sigma1 * np.exp(-d2/2/(sigma1**2)) - 1/sigma2 * np.exp(-d2/2/(sigma2**2)))
    gfilter = gfilter - np.mean(gfilter)
    gfilter = gfilter / np.max(gfilter)
    return gfilter

def feature_visualization(features, plotx=8,ploty=8,layer_num=[2,3,4],filter_sizes=[5,2,3],\
                             filter_strides=[1,2,1],nof_filters=[32,32,64],types=['conv','pool','conv'],currLayer=4,\
                             show=True, figsize=(10,10), font1=12, font2=18):
    feature_sizes=[]
    layer_weights = [items[0] for items in features]
    feature_size=0
    index = layer_num.index(currLayer)
    temp =filter_sizes[index]
    for index_ in range(index,0,-1):
        feature_size = (temp*filter_strides[index_-1]+filter_sizes[index_-1])
        feature_sizes.insert(0,feature_size)
        temp = feature_size
    layer = layer_num[index]
    layer_weight_index = (int(layer/2))-1
    conv_features = layer_weights[layer_weight_index]
    for index_ in range(index,0,-1):
        if(types[index_-1]=='pool'):
            pool_features = np.zeros((feature_sizes[index_-1],\
                feature_sizes[index_-1],nof_filters[index_-1],nof_filters[index]))
            for axis_ in range(nof_filters[index]):
                a_pool_feature = np.zeros((feature_sizes[index_-1],\
                               feature_sizes[index_-1],nof_filters[index_-1]))
                #print(index_, axis_, conv_features.shape, layer_weight_index)
                a_conv_feature = conv_features[:,:,:,axis_]
                locs = np.where(a_conv_feature>0.05)
                pool_locs = [filter_strides[index_-1]*items for items in locs[0:2]]
                pool_locs.append(locs[-1])
                pool_locs = tuple(pool_locs)
                a_pool_feature[pool_locs]=a_conv_feature[locs]
                pool_features[:,:,:,axis_]=a_pool_feature
        elif(types[index_-1]=='conv'):
            if(index_==1):
                final_features = np.zeros((feature_sizes[index_-1],\
                                   feature_sizes[index_-1],3,nof_filters[index]))
                for axis_ in range(nof_filters[index]):
                    for iii in range(pool_features[:,:,:,axis_].shape[0]):
                        for jjj in range(pool_features[:,:,:,axis_].shape[1]):
                            mxv=pool_features[iii,jjj,:,axis_].max()
                            mxi=pool_features[iii,jjj,:,axis_].argmax()
                            if(mxv>0.3):
                                strd=filter_strides[index_-1]
                                szs=filter_sizes[index_-1]
                                #final_features[(iii-1)*strd+1:(iii)*strd+szs,\
        #(jjj-1)*strd+1:(jjj)*1+szs,0,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[index_-1][:,:,1,mxi]
                                final_features[(iii-1)*strd+1:(iii)*strd+szs,\
        (jjj-1)*strd+1:(jjj)*1+szs,1,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[index_-1][:,:,0,mxi]
            else:
                conv_features = np.zeros((feature_sizes[index_-1],\
                                   feature_sizes[index_-1],nof_filters[index_-2],nof_filters[index_+1]))
                for axis_ in range(nof_filters[index]):
                    for iii in range(pool_features[:,:,:,axis_].shape[0]):
                        for jjj in range(pool_features[:,:,:,axis_].shape[1]):
                            mxv=pool_features[iii,jjj,:,axis_].max()
                            mxi=pool_features[iii,jjj,:,axis_].argmax()
                            if(mxv>0.3):
                                strd=filter_strides[index_-1]
                                szs=filter_sizes[index_-1]
                                layer = layer_num[index_]
                                layer_weight_index = (layer/2)-1
                                conv_features[(iii-1)*strd+1:(iii)*strd+szs,\
        (jjj-1)*strd+1:(jjj)*1+szs,:,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[layer_weight_index][:,:,:,mxi]

    final_features = final_features
    if(show):
        fig, axes = plt.subplots(plotx, ploty,figsize=figsize,
                     subplot_kw={'xticks': [], 'yticks': []})
        #fig.subplots_adjust(left=0.03, bottom=0.0, right=0.99, top=0.9, wspace=0.27, hspace=0.21)      
        axes = axes.flat
        for i in range(len(axes)):
            axes[i].imshow(final_features[:,:,1,i],interpolation='none', cmap='gray')
            axes[i].set_title('Map'+str(i+1),fontsize=font1)
        #plt.show()
    
        return fig, final_features
