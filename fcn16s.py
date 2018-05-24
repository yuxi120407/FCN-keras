# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:27:17 2018

@author: yuxi
"""

#import matplotlib.pyplot as plt
#from PIL import Image
#
#import numpy as np
#import math
#import copy
#
#import skimage.io as io
#from scipy.misc import bytescale
#from keras.models import Sequential, Model
#from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
#from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
#from keras.layers import merge
#from unit import fcn32_blank, fcn_32s_to_16s, prediction
##%%
#image_size = 64*8 # INFO: initially tested with 256, 448, 512
#fcn32model = fcn32_blank(image_size)
#
##fcn16model = fcn_32s_to_16s(fcn32model)
##imarr = np.ones((3,image_size,image_size))
##imarr = np.expand_dims(imarr, axis=0)
##
###testmdl = Model(fcn32model.input, fcn32model.layers[10].output) # works fine
##testmdl = fcn16model # works fine
##testmdl.predict(imarr).shape
import copy
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Permute, Add, add
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import merge
from scipy.misc import bytescale

def convblock(cdim, nb, bits=3):
	L = []

	for k in range(1, bits + 1):
		convname = 'conv' + str(nb) + '_' + str(k)
		if False:
			# first version I tried
			L.append(ZeroPadding2D((1, 1)))
			L.append(Convolution2D(cdim, kernel_size=(3, 3), activation='relu', name=convname))
		else:
			L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))

	L.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	return L


def fcn32_blank(image_size=512):
	withDO = False  # no effect during evaluation but usefull for fine-tuning

	if True:
		mdl = Sequential()

		# First layer is a dummy-permutation = Identity to specify input shape
		mdl.add(Permute((1, 2, 3), input_shape=(image_size, image_size, 3)))  # WARNING : axis 0 is the sample dim

		for l in convblock(64, 1, bits=2):
			mdl.add(l)

		for l in convblock(128, 2, bits=2):
			mdl.add(l)

		for l in convblock(256, 3, bits=3):
			mdl.add(l)

		for l in convblock(512, 4, bits=3):
			mdl.add(l)

		for l in convblock(512, 5, bits=3):
			mdl.add(l)

		mdl.add(Convolution2D(4096, kernel_size=(7, 7), padding='same', activation='relu', name='fc6'))  # WARNING border
		if withDO:
			mdl.add(Dropout(0.5))
		mdl.add(Convolution2D(4096, kernel_size=(1, 1), padding='same', activation='relu', name='fc7'))  # WARNING border
		if withDO:
			mdl.add(Dropout(0.5))

		# WARNING : model decapitation i.e. remove the classifier step of VGG16 (usually named fc8)

		mdl.add(Convolution2D(21, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr'))

		convsize = mdl.layers[-1].output_shape[2]
		deconv_output_size = (convsize - 1) * 2 + 4  # INFO: =34 when images are 512x512
		# WARNING : valid, same or full ?
		mdl.add(Deconvolution2D(21, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation=None, name='score2'))

		extra_margin = deconv_output_size - convsize * 2  # INFO: =2 when images are 512x512
		assert (extra_margin > 0)
		assert (extra_margin % 2 == 0)
		# INFO : cropping as deconv gained pixels
		# print(extra_margin)
		c = ((0, extra_margin), (0, extra_margin))
		# print(c)
		mdl.add(Cropping2D(cropping=c))
		# print(mdl.summary())

		return mdl

	else:
		# See following link for a version based on Keras functional API :
		# gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
		raise ValueError('not implemented')


# WARNING : explanation about Deconvolution2D layer
# http://stackoverflow.com/questions/39018767/deconvolution2d-layer-in-keras
# the code example in the help (??Deconvolution2D) is very usefull too
# ?? Deconvolution2D

def fcn_32s_to_16s(fcn32model=None):
	if fcn32model is None:
		fcn32model = fcn32_blank()

	fcn32shape = fcn32model.layers[-1].output_shape
	assert (len(fcn32shape) == 4)
	assert (fcn32shape[0] is None)  # batch axis
	assert (fcn32shape[3] == 21)  # number of filters
	assert (fcn32shape[1] == fcn32shape[2])  # must be square

	fcn32size = fcn32shape[1]  # INFO: =32 when images are 512x512

	if fcn32size != 32:
		print('WARNING : handling of image size different from 512x512 has not been tested')

	sp4 = Convolution2D(21, kernel_size=(1, 1), padding='same', activation=None, name='score_pool4')

	# INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
	# https://keras.io/getting-started/sequential-model-guide/
	summed = add(inputs=[sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output])

	# INFO :
	# final 16x16 upsampling of "summed" using deconv layer upsample_new (32, 32, 21, 21)
	# deconv setting is valid if (528-32)/16 + 1 = deconv_input_dim (= fcn32size)
	deconv_output_size = (fcn32size - 1) * 16 + 32  # INFO: =528 when images are 512x512
	upnew = Deconvolution2D(21, kernel_size=(32, 32),
							padding='valid',  # WARNING : valid, same or full ?
							strides=(16, 16),
							activation=None,
							name='upsample_new')

	extra_margin = deconv_output_size - fcn32size * 16  # INFO: =16 when images are 512x512
	assert (extra_margin > 0)
	assert (extra_margin % 2 == 0)
	# print(extra_margin)
	# INFO : cropping as deconv gained pixels
	crop_margin = Cropping2D(cropping=((0, extra_margin), (0, extra_margin)))

	return Model(fcn32model.input, crop_margin(upnew(summed)))

def fcn_32s_to_8s(fcn32model=None):
    
    if (fcn32model is None):
        fcn32model = fcn32_blank()
        
    fcn32shape = fcn32model.layers[-1].output_shape
    assert (len(fcn32shape) == 4)
    assert (fcn32shape[0] is None) # batch axis
    assert (fcn32shape[3] == 21) # number of filters
    assert (fcn32shape[1] == fcn32shape[2]) # must be square
    
    fcn32size = fcn32shape[1] # INFO: =32 when images are 512x512
    
    if (fcn32size != 32):
        print('WARNING : handling of image size different from 512x512 has not been tested')
    
    sp4 = Convolution2D(21, kernel_size=(1, 1), padding='same', activation=None, name='score_pool4')

    # INFO : to replicate MatConvNet.DAGN.Sum layer see documentation at :
    # https://keras.io/getting-started/sequential-model-guide/
    summed = add(inputs=[sp4(fcn32model.layers[14].output), fcn32model.layers[-1].output])


    deconv4_output_size = (fcn32size-1)*2+4 # INFO: =66 when images are 512x512
    s4deconv = Deconvolution2D(21,kernel_size=( 4, 4),
                            #output_shape=(None, 21, deconv4_output_size, deconv4_output_size),
                            padding='valid', # WARNING : valid, same or full ?
                            strides=(2, 2),
                            activation=None,
                            name = 'score4')

    extra_margin4 = deconv4_output_size - fcn32size*2 # INFO: =2 when images are 512x512
    assert (extra_margin4 > 0)
    assert (extra_margin4 % 2 == 0)
    crop_margin4 = Cropping2D(cropping=((0, extra_margin4), (0, extra_margin4))) # INFO : cropping as deconv gained pixels

    score4 = crop_margin4(s4deconv(summed))

    # WARNING : check dimensions
    sp3 = Convolution2D(21, kernel_size=(1, 1),
                        padding='same', # WARNING : zero or same ? does not matter for 1x1
                        activation=None, # WARNING : to check
                        name='score_pool3')

    score_final = add(inputs=[sp3(fcn32model.layers[10].output), score4]) # WARNING : is that correct ?

    assert (fcn32size*2 == fcn32model.layers[10].output_shape[1])
    deconvUP_output_size = (fcn32size*2-1)*8+16 # INFO: =520 when images are 512x512
    upsample = Deconvolution2D(21, kernel_size=(16, 16),
                            #output_shape=(None, 21, deconvUP_output_size, deconvUP_output_size),
                            padding='valid', # WARNING : valid, same or full ?
                            strides=(8, 8),
                            activation=None,
                            name = 'upsample')

    bigscore = upsample(score_final)

    extra_marginUP = deconvUP_output_size - (fcn32size*2)*8 # INFO: =8 when images are 512x512
    assert (extra_marginUP > 0)
    assert (extra_marginUP % 2 == 0)
    crop_marginUP = Cropping2D(cropping=((0, extra_marginUP), (0, extra_marginUP)))# INFO : cropping as deconv gained pixels

    coarse = crop_marginUP(bigscore)

    return Model(fcn32model.input, coarse)

def prediction(kmodel, crpimg, transform=False):
	# INFO : crpimg should be a cropped image of the right dimension

	# transform=True seems more robust but I think the RGB channels are not in right order

	imarr = np.array(crpimg).astype(np.float32)

	if transform:
		imarr[:, :, 0] -= 129.1863
		imarr[:, :, 1] -= 104.7624
		imarr[:, :, 2] -= 93.5940
		#
		# WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
		aux = copy.copy(imarr)
		imarr[:, :, 0] = aux[:, :, 2]
		imarr[:, :, 2] = aux[:, :, 0]

	# imarr[:,:,0] -= 129.1863
	# imarr[:,:,1] -= 104.7624
	# imarr[:,:,2] -= 93.5940

	# imarr = imarr.transpose((2, 0, 1))
	imarr = np.expand_dims(imarr, axis=0)

	return kmodel.predict(imarr)

#%%
if __name__ == "__main__":
	md = fcn32_blank()
	md = fcn_32s_to_8s(md)
	print(md.summary())
#%%
data = loadmat('pascal-fcn8s-dag.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
p = data['params']
description = data['meta'][0,0].classes[0,0].description
#%%
l.shape, p.shape, description.shape
#%%
class2index = {}
for i, clname in enumerate(description[0,:]):
    class2index[str(clname[0])] = i
    
print(sorted(class2index.keys()))
if False: # inspection of data structure
    print(dir(l[0,31].block[0,0]))
    print(dir(l[0,36].block[0,0]))
for i in range(0, p.shape[1]-1, 2):
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape,
          str(p[0,i+1].name[0]), p[0,i+1].value.shape)

for i in range(l.shape[1]):
    print(i,
          str(l[0,i].name[0]), str(l[0,i].type[0]),
          [str(n[0]) for n in l[0,i].inputs[0,:]],
          [str(n[0]) for n in l[0,i].outputs[0,:]])
    
#%%
def copy_mat_to_keras(kmodel):
    
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (0, 1, 2, 3) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, p.shape[1]-1, 2):
        matname = '_'.join(p[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print('found : ', (str(matname), kindex))
            l_weights = p[0,i].value
            l_bias = p[0,i+1].value
            f_l_weights = l_weights.transpose(prmt)
            if False: # WARNING : this depends on "image_data_format":"channels_last" in keras.json file
                f_l_weights = np.flip(f_l_weights, 0)
                f_l_weights = np.flip(f_l_weights, 1)
            print(f_l_weights.shape, kmodel.layers[kindex].get_weights()[0].shape)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
        else:
            print('not found : ', str(matname))
    
#%%
copy_mat_to_keras(md)    
#%%201208172_T-12-58-58_Dive_01_041
im = Image.open('rgb.jpg') # http://www.robots.ox.ac.uk/~szheng/crfasrnndemo/static/rgb.jpg
#im = Image.open('201208172_T-12-58-58_Dive_01_041.jpg')
#im = im.crop((0,0,319,319)) # WARNING : manual square cropping
im = im.resize((512,512))
#%%
plt.imshow(np.asarray(im))
print(np.asarray(im).shape)    
#%%
crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit
preds = prediction(md, crpim, transform=False) # WARNING : transfrom=True requires a code change (dim order)    
#%%
print(preds.shape)
imclass = np.argmax(preds, axis=3)[0,:,:]
print(imclass.shape)
plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow( np.asarray(crpim) )
plt.subplot(1,3,2)
plt.imshow( imclass )
plt.subplot(1,3,3)
plt.imshow( np.asarray(crpim) )
masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#plt.imshow( imclass, alpha=0.5 )
plt.imshow( masked_imclass, alpha=0.5 )   
#%%
# List of dominant classes found in the image
for c in np.unique(imclass):
    print(c, str(description[0,c][0]))    
#%%
bspreds = bytescale(preds, low=0, high=255)

plt.figure(figsize = (15, 7))
plt.subplot(2,3,1)
plt.imshow(np.asarray(crpim))
plt.subplot(2,3,3+1)
plt.imshow(bspreds[0,:,:,class2index['background']], cmap='seismic')
plt.subplot(2,3,3+2)
plt.imshow(bspreds[0,:,:,class2index['person']], cmap='seismic')
plt.subplot(2,3,3+3)
plt.imshow(bspreds[0,:,:,class2index['bicycle']], cmap='seismic')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
