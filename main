from model_1 import *
from data_1 import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,r'C:\Users\sen\Desktop\unet_1\data\train','image','mask',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=30,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator(r"C:\Users\sen\Desktop\unet_1\data\test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult(r"C:\Users\sen\Desktop\unet_1\data\test",results)
