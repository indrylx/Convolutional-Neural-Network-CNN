from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os
import random
from shutil import copyfile

#Directory
base_dir = 'PATH'
#menentukan direktori
bahan_dir = os.path.join(base_dir, 'DataSet')
train_dir = os.path.join(base_dir, 'DataTrain')
validation_dir = os.path.join(base_dir, 'DataTest')
#menentukan direktori isi bahan
cos_dir = os.path.join(bahan_dir,'Cos/')
div_dir = os.path.join(bahan_dir,'Div/')
pi_dir = os.path.join(bahan_dir,'Pi/')
plus_dir = os.path.join(bahan_dir,'Plus/')
sin_dir = os.path.join(bahan_dir,'Sin/')
substract_dir = os.path.join(bahan_dir,'Substract/')
tan_dir = os.path.join(bahan_dir,'Tan/')
x_dir = os.path.join(bahan_dir,'X/')

print("Jumlah Data ")
print('Jumlah Data Gambar Cos : ' ,len(os.listdir(cos_dir)))
print('Jumlah Data Gambar Div : ' ,len(os.listdir(div_dir)))
print('Jumlah Data Gambar Pi : ' ,len(os.listdir(pi_dir)))
print('Jumlah Data Gambar Plus : ' ,len(os.listdir(plus_dir)))
print('Jumlah Data Gambar Sin : ' ,len(os.listdir(sin_dir)))
print('Jumlah Data Gambar Substract : ' ,len(os.listdir(substract_dir)))
print('Jumlah Data Gambar Tan : ' ,len(os.listdir(tan_dir)))
print('Jumlah Data Gambar X : ' ,len(os.listdir(x_dir)))

#direktori isi train
train_cos = os.path.join(train_dir,'Cos/')
train_div = os.path.join(train_dir,'Div/')
train_pi = os.path.join(train_dir,'Pi/')
train_plus = os.path.join(train_dir,'Plus/')
train_sin = os.path.join(train_dir,'Sin/')
train_substract = os.path.join(train_dir,'Substract/')
train_tan = os.path.join(train_dir,'Tan/')
train_x = os.path.join(train_dir,'X/')

#direktori isi validasi/test
validation_cos = os.path.join(validation_dir,'Cos/')
validation_div = os.path.join(validation_dir,'Div/')
validation_pi = os.path.join(validation_dir,'Pi/')
validation_plus = os.path.join(validation_dir,'Plus/')
validation_sin = os.path.join(validation_dir,'Sin/')
validation_substract = os.path.join(validation_dir,'Substract/')
validation_tan = os.path.join(validation_dir,'Tan/')
validation_x = os.path.join(validation_dir,'X/')



print('jumlah All Data Gambar Cos', len(os.listdir(cos_dir)))
print('jumlah Train Data Gambar Cos', len(os.listdir(train_cos)))
print('jumlah Val Data Gambar Cos', len(os.listdir(validation_cos)))

print('jumlah All Data Gambar Div', len(os.listdir(div_dir)))
print('jumlah Train Data Gambar Div', len(os.listdir(train_div)))
print('jumlah Val Data Gambar Div', len(os.listdir(validation_div)))

print('jumlah All Data Gambar Pi', len(os.listdir(pi_dir)))
print('jumlah Train Data Gambar Pi', len(os.listdir(train_dir)))
print('jumlah Val Data Gambar Pi', len(os.listdir(validation_pi)))

print('jumlah All Data Gambar Plus', len(os.listdir(plus_dir)))
print('jumlah Train Data Gambar Plus', len(os.listdir(train_plus)))
print('jumlah Val Data Gambar Plus', len(os.listdir(validation_plus)))

print('jumlah All Data Gambar Sin', len(os.listdir(sin_dir)))
print('jumlah Train Data Gambar Sin', len(os.listdir(train_sin)))
print('jumlah Val Data Gambar Sin', len(os.listdir(validation_sin)))

print('jumlah All Data Gambar Substract', len(os.listdir(substract_dir)))
print('jumlah Train Data Gambar Substract', len(os.listdir(train_substract)))
print('jumlah Val Data Gambar Substract', len(os.listdir(validation_substract)))

print('jumlah All Data Gambar Tan', len(os.listdir(tan_dir)))
print('jumlah Train Data Gambar Tan', len(os.listdir(train_tan)))
print('jumlah Val Data Gambar Tan', len(os.listdir(validation_tan)))

print('jumlah All Data Gambar X', len(os.listdir(x_dir)))
print('jumlah Train Data Gambar X', len(os.listdir(train_x)))
print('jumlah Val Data Gambar X', len(os.listdir(validation_x)))



classifier = Sequential()
classifier.add(Conv2D(32, 3, padding='same', activation ='relu'))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(64, 3, padding='same', activation ='relu'))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(64, 3, padding='same', activation ='relu'))
classifier.add(MaxPooling2D())
classifier.add(Dropout(0.3))
classifier.add(Flatten())
classifier.add(Dense(64, activation ='relu'))
classifier.add(Dense(64, activation ='relu'))
classifier.add(Dense(8, activation ='softmax'))
classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory("PATH",
target_size = (64, 64),
batch_size = 50,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory("PATH",
target_size = (64, 64),
batch_size = 50,
class_mode = 'categorical')
classifier.fit(training_set,
steps_per_epoch = len(training_set),
epochs = 50,
validation_data = test_set,
validation_steps = len(test_set)) 

import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
test_image = image.load_img("D:/ML/TugasML/DataSet/X/exp63.jpg", target_size = (64, 64))
imgplot = plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
classes = np.argmax(result)
print(training_set.class_indices)
print(classes)
 
if classes==0:
    print('Ini Gambar Cos')
    plt.title('Gambar Ini Simbol Cos')
    plt.show()
elif classes==1:
    print('Ini Gambar Div')
    plt.title('Gambar Ini Simbol Div')
    plt.show()
elif classes==2:
    print('Ini Gambar Pi')
    plt.title('Gambar Ini Simbol Pi')
    plt.show()
elif classes==3:
    print('Ini Gambar Plus')
    plt.title('Gambar Ini Simbol Plus')
    plt.show()
elif classes==4:
    print('Ini Gambar Sin')
    plt.title('Gambar Ini Simbol Sin')
    plt.show()
elif classes==5:
    print('Ini Gambar Substract')
    plt.title('Gambar Ini Simbol Substract')
    plt.show()
elif classes==6:
    print('Ini Gambar Tan')
    plt.title('Gambar Ini Simbol Tan')
    plt.show()
elif classes==7:
    print('Ini Gambar X')
    plt.title('Gambar Ini Simbol X')
    plt.show()
else:
    print('Gambar Tidak Ada Di Dataset')