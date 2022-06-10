from model import *
from data import *

import sys
import os
import time

# 当训练检测倾斜鱼体是才开启rotation_range
data_gen_args = dict(
                    rotation_range=45,
                    width_shift_range=0.02,
                    height_shift_range=0.02,
                    zoom_range=0.01,
                    fill_mode='nearest')
#myGene = trainGenerator(1,'data/mydata/train','image','label',data_gen_args,save_to_dir = "data/mydata/train/aug",target_size = (864,1152))
myGene = trainGenerator(2,'data/mydata/train','image','label',data_gen_args,save_to_dir = None,target_size = (864,1152))

#model = unet(pretrained_weights = 'unet_mydata.hdf5' ,input_size = (864,1152,1))
model = unet(input_size = (864,1152,1))
model_checkpoint = ModelCheckpoint('unet_mydata.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=600,epochs=50,callbacks=[model_checkpoint])

testGene = testGenerator("data/mydata/test",target_size = (864,1152))
results = model.predict_generator(testGene,50,verbose=1)
saveResult("data/mydata/test_results",results)



