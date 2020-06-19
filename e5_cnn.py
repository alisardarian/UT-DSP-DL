#!/usr/bin/env python
# coding: utf-8

# # DSP HW #5<img src="https://upload.wikimedia.org/wikipedia/en/thumb/f/fd/University_of_Tehran_logo.svg/1200px-University_of_Tehran_logo.svg.png" width=120px style="float: right;"/>
# ## Ali Sardarian
# ### UT 2020

# #### :کتاب‌خانه‌های مورد نیاز

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import thinkdsp as dsp
import glob


# In[2]:


# Check GPU Availability:
print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))


# ### :CIFAR-10 پیش پردازش داده‌های آموزش 

# مجموعه داده‌ها از حافظه‌ی محلی خوانده می‌شوند، در صورت اجرای آنلاین می‌توان از دستور زیر نیز استفاده کرد
# 
# (x_train, y_train) = tf.keras.datasets.cifar10.load_data()
# 
# https://www.tensorflow.org/datasets/catalog/cifar10

# In[3]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[4]:


dataset_batches = 5
training_inputs_list = []
training_labels_list = []
for i in range(dataset_batches):
    data_dict_1 = unpickle("cifar-10-batches-py\\data_batch_"+str(i+1))
    training_inputs_list.extend(data_dict_1[b'data'])
    training_labels_list.extend(data_dict_1[b'labels'])
training_inputs = np.array(training_inputs_list)
# nomalization: (0->1)
training_inputs = training_inputs.reshape((dataset_batches*10000, 32, 32, 3),order='F').swapaxes(1,2)/255.0
# convert labels to one-hot:
training_labels = tf.one_hot(training_labels_list, 10).numpy()
print ("training data shape : ", training_inputs.shape)
print ("training labels shape : ", training_labels.shape)


# ### :CIFAR-10 پیش پردازش داده‌های آزمون 

# In[5]:


testing_inputs_list = []
testing_labels_list = []
data_dict_2 = unpickle("cifar-10-batches-py\\test_batch")
testing_inputs_list.extend(data_dict_2[b'data'])
testing_labels_list.extend(data_dict_2[b'labels'])
testing_inputs = np.array(testing_inputs_list)
# nomalization: (0->1)
testing_inputs = testing_inputs.reshape((10000, 32, 32, 3),order='F').swapaxes(1,2)/255.0
testing_labels = tf.one_hot(testing_labels_list, 10).numpy() # convert labels to one-hot
print ("testing data shape : ", testing_inputs.shape)
print ("testing labels shape : ", testing_labels.shape)


# #### بعد از خواندن داده‌ها از حافظه، مقدار داده‌های آموزش نرمال می‌شوند و همچنین برای داده‌های برچسب نیز آرایه وان هات تولید می‌شود

# ### :CIFAR-10 برچسب داده‌های  

# In[6]:


label_names = {}
data_dict_3 = unpickle("cifar-10-batches-py\\batches.meta")
label_names = data_dict_3[b'label_names']
print(label_names)


# ### :CIFAR-10 نمایش نمونه داده‌های  

# In[7]:


plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(training_inputs[np.random.randint(0, training_inputs.shape[0]-1)])
    plt.axis('off')
plt.show()


# ## :تعیین معماری مدل پایه

# #### :تعیین لایه‌های شبکه و پارامتر‌های آن

# In[8]:


model_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=7, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=9, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])


# #### :تعیین تابع بهینه‌ساز و خطای مناسب شبکه

# In[9]:


model_1.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=0.01),
            loss= tf.keras.losses.CategoricalCrossentropy(),
            metrics= [tf.keras.metrics.CategoricalAccuracy()]
        )


# #### هدف از پیاده‌سازی کلاس کال بک زیر، ذخیره‌ی مقدار‌های خطا در پایان هر تکرار و همچنین محاسبه‌ی زمان هر تکرار است و در نهایت با اتمام فرآیند آموزش، نمودارهای روند تغییرات خطا بر روی داده‌ها و مقدار میانگین زمان هر تکرار را در خروجی نشان می‌دهد. از این کال بک در آموزش تمام مدل‌ها در ادامه استفاده خواهد شد 

# In[10]:


class MyCustomCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epochTimes = []
        self.trainLoss = []
        self.validationLoss = []
    def on_train_begin(self,logs = {}):
        self.lastEpochTime = tf.timestamp().numpy() # beginning of training time
    def on_epoch_end(self,epoch,logs = {}):
        self.epochTimes.append(tf.timestamp().numpy() - self.lastEpochTime) # calculate time of this epoch
        self.lastEpochTime = tf.timestamp().numpy() # update for next epoch
        self.trainLoss.append(logs['loss']) # record this epoch train error
        self.validationLoss.append(logs['val_loss']) # record this epoch validation error
    def on_train_end(self,logs = {}):
        # print the average of epochs time:
        print ("\n\n\n\n\nAverage Time of Epochs =", np.mean(np.array(self.epochTimes)), "s")
        # draw error diagram of model:
        plt.plot(self.trainLoss, color='blue', label='train error')
        plt.plot(self.validationLoss, color='red', label='validation error', linestyle='--')
        plt.xlabel('epochs') 
        plt.ylabel('error') 
        plt.title("Notwork's Error Diagram")
        plt.legend(loc=0)
        plt.show()


# # :قسمت الف

# ### :آموزش مدل 

# در آموزش این مدل از داده‌های آموزشی آماده شده به همراه برچسب آن‌ها استفاده می‌شود، همچنین با تعیین پارامتر تقسیم داده‌های ولیدیشن، 20 درصد از داده‌های آموزش به عنوان داده‌های ولیدیشن جدا می‌شوند، به منظور توقف زودهنگام فرآیند آموزش در صورت افزایش متوالی مقدار خطای ولیدیشن، از کال بک آماده مخصوص آن استفاده شده تا در صورت افزایش خطای ولیدیشن در 3 مرحله‌ی متوالی، فرآیند آموزش متوقف گردد 

# In[11]:


history_1 = model_1.fit(
                x= training_inputs,
                y= training_labels,
                batch_size= 32,
                epochs= 100,
                verbose= 2,
                callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                           MyCustomCallBack()],
                validation_split= 0.2,
                shuffle=True
            )


# ### :ارزیابی مدل  

# In[12]:


test_result_1 = model_1.evaluate(
    x= testing_inputs,
    y= testing_labels,
    verbose= 2
    )
print ("\n\nAccuracy on CIFAR-10 Test Data =", test_result_1[1]*100,"%")


# مقدار دقت این مدل بر روی داده‌های آزمایشی در بازه‌ی 40 تا  50 درصد است

# # :قسمت ب

# ### :پیش پردازش داده‌های آموزش اعداد 

# در این مرحله، سیگنال‌های اعداد ابتدا از حافظه خوانده می‌شوند و سپس با کمک یک تابع آماده از کتابخانه‌های پیشفرض پایتون، به تصویر اسپکتروگرام تبدیل می‌شوند، به منظور رنگی بودن تصویر تولیدی از کالرمپ مخصوصی استفاده شده که بنظر مناسب است و هر سه رنگ آبی، قرمز و سبز را پوشش می‌دهد، اسپکتروگرام‌های تولید شده در یک لیست ذخیره شده و در نهایت عملیات نرمال سازی و مخلوط‌سازی بر روی آن‌ها انجام می‌شود. به دلیل خواندن ترتیبی داده‌ها استفاده از مخلوط‌سازی به صورت تصادفی برای داشتن دقت بهتر ضروری است
# 
# همچنین به دلیل ورودی ثابت لایه‌ی پیچشی مدل، با استفاده از دستور ریسایز کتاب‌خانه تنسورفلو، هر اسپکتروگرام تغییر اندازه داده می‌شود
# 
# نمونه‌ای از اسپکتروگرام‌های به دست آمده در ادامه قابل مشاهده است

# In[13]:


training_inputs_list_2 = []
training_labels_list_2 = []


for i in range(10):
    files = glob.glob("SpeechRecognition\\TrainSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave = dsp.read_wave(f)
        wave_array = wave.ys
        spectrum, freqs, t, im = plt.specgram(x=wave_array, Fs=wave.framerate, cmap='jet') # make spectrogram from wave samples
        renderer = plt.gcf().canvas.get_renderer() # current renderer handle
        image = im.make_image(renderer)[0][:,:,:-1] # make image of spectrogram
        plt.close()
        training_inputs_list_2.append(image)
        training_labels_list_2.append(tf.one_hot(i, 10).numpy()) # labeling
        
# shuffle data:
combined_list = list(zip(training_inputs_list_2, training_labels_list_2))
np.random.shuffle(combined_list)
training_inputs_2, training_labels_2 = zip(*combined_list)
training_inputs_2 = np.array(training_inputs_2)
# nomalization: (0->1)
training_inputs_2 = training_inputs_2/255.0
# resize: (32x32)
with tf.device('/CPU:0'):
    training_inputs_2 = tf.image.resize(training_inputs_2, [32,32]).numpy()
training_labels_2 = np.array(training_labels_2)
print ("training data shape : ", training_inputs_2.shape)
print ("training labels shape : ", training_labels_2.shape)


# ### :پیش پردازش داده‌های آزمون اعداد 

# In[14]:


testing_inputs_list_2 = []
testing_labels_list_2 = []
testing_inputs_waves = []

for i in range(10):
    files = glob.glob("SpeechRecognition\\TestSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave = dsp.read_wave(f)
        testing_inputs_waves.append(wave)
        wave_array = wave.ys
        spectrum, freqs, t, im = plt.specgram(x=wave_array, Fs=wave.framerate, cmap='jet') # make spectrogram from wave samples
        renderer = plt.gcf().canvas.get_renderer() # current renderer handle
        image = im.make_image(renderer)[0][:,:,:-1] # make image of spectrogram
        plt.close()
        testing_inputs_list_2.append(image)
        testing_labels_list_2.append(tf.one_hot(i, 10).numpy()) # labeling
        
testing_inputs_2 = np.array(testing_inputs_list_2)
# nomalization: (0->1)
testing_inputs_2 = testing_inputs_2/255.0
# resize: (32x32)
with tf.device('/CPU:0'):
    testing_inputs_2 = tf.image.resize(testing_inputs_2, [32,32]).numpy()
testing_labels_2 = np.array(testing_labels_list_2)
print ("testing data shape : ", testing_inputs_2.shape)
print ("testing labels shape : ", testing_labels_2.shape)


# ### :نمایش نمونه داده‌های اعداد 

# In[15]:


plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(training_inputs_2[i])
    plt.axis('off')
plt.show()


# ### :پیش آموزش مدل بازشناسی اعداد 

# در این مرحله شبکه‌ای جدید شامل تمام لایه‌های شبکه پایه قبل به غیر از لایه‌ی آخر آن ساخته می‌شود و سپس یک لایه‌ی خطی جدید به آن اضافه می‌شود

# In[16]:


model_2 = tf.keras.Sequential(name='numbers_cnn')
for layer in model_1.layers[:-1]: # exclude final layer
    model_2.add(layer)
model_2.add(tf.keras.layers.Dense(units=10, activation='softmax', name='new_final_layer')) # add a new final layer
model_2.summary()


# #### :تعیین تابع بهینه‌ساز و خطای مناسب شبکه

# In[17]:


model_2.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=0.01),
            loss= tf.keras.losses.CategoricalCrossentropy(),
            metrics= [tf.keras.metrics.CategoricalAccuracy()]
        )


# ### :بازآموزش مدل بازشناسی اعداد 

# مدل ساخته شده که شامل وزن‌های شبکه‌ی قبل است با لایه آخر جدید و داده‌های اعداد آموزش داده می‌شود

# In[18]:


history_2 = model_2.fit(
                x= training_inputs_2,
                y= training_labels_2,
                batch_size= 32,
                epochs= 100,
                verbose= 2,
                callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                           MyCustomCallBack()],
                validation_split= 0.2,
                shuffle=True
            )


# ### :ارزیابی مدل بازشناسی اعداد 

# In[19]:


test_result_2 = model_2.evaluate(
    x= testing_inputs_2,
    y= testing_labels_2,
    verbose= 2
    )
print ("\n\nAccuracy on Numbers Test Data =", test_result_2[1]*100,"%")


# مقدار دقت این مدل بر روی داده‌های آزمون اعداد تقریبا 90 درصد است که مقدار مناسبی است

# # :قسمت پ

# ### :CIFAR-10 انتخاب یک نمونه از مجموعه داده  

# In[20]:


# select a random test input from CIFAR-10:
selected_image = testing_inputs[6325]
selected_image = np.expand_dims(selected_image, axis=0)
plt.figure(figsize=(2,2))
plt.imshow(selected_image[0])
plt.axis('off')
plt.title('Selected Image:') 
plt.show()


# In[21]:


print("output label =",label_names[np.argmax(model_1.predict(selected_image))]) # TEST


# ### :دریافت خروجی و فیلتر‌های لایه اول پیچشی  

# In[22]:


conv2d_outputs = model_2.layers[0](selected_image).numpy()[0]
conv2d_outputs = conv2d_outputs/conv2d_outputs.max() # normalize
print("Conv2D outputs shape:",conv2d_outputs.shape)
conv2d_kernels = model_2.layers[0].get_weights()[0]
conv2d_kernels = conv2d_kernels/conv2d_kernels.max() # normalize
print("Conv2D kernels shape:",conv2d_kernels.shape)


# ### :نمایش تصویر کانال‌های لایه اول در کنار تصویر فیلتر‌های هر کانال 

# In[23]:


fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(10, 15))
for i, ax in enumerate(axes.flat):
    if (i%4 == 0):
        # output image:
        ax.imshow(conv2d_outputs[:,:,int(i/4)], cmap = 'gray', interpolation='nearest')
        ax.set_title("output "+str(int(i/4)+1))
    else:
        # kernel image:
        ax.imshow(conv2d_kernels[:,:,(i%4)-1,int(i/4)], cmap = 'gray', interpolation='nearest')
        ax.set_title("kernel "+str(int(i/4)+1)+" (channel "+str((i%4))+")")
    ax.axis('off')
plt.show()


# ### :نتیجه 

# در این بخش خروجی‌های لایه اول پیچشی نمایش داده است، تعداد این تصاویر خروجی هم اندازه‌ی تعداد فیلترها است که فیلتر هر کانال خروجی در همان ردیف رسم شده است و کانال‌های رنگی هر فیلتر به صورت جدا از هم نشان داده شده‌اند. از بین 7 کانال خروجی فقط سه کانال داری تصویر بوده و قابل تفسیر هستند
# 
# می‌توان اینطور نتیجه گرفت که در خروجی شماره 5، تمرکز بر روی قسمت‌های برجسته تصویر اصلی یا فورگراند و نواحی روشن تر است، زیر در خروجی این نواحی نقاط روشن‌تری داریم. در مقایسه با تصویر‌های کرنل این کانال، می‌توان دید که هر سه نقاط مرکزی را عبور می‌دهند و چیدمان قطری دارند ولی در خروجی شماره 6 می‌توان دید که تمرکز بر روی ناحیه‌های عقب‌تر و عمیق تصویر یا بک گراند است و فیلتر‌های آن نیز بیشتر نقاط مرکزی را از خود عبور داده و چیدمان خطی دارند
# 
# در خروجی شماره 4 نیز اغلب قسمت‌های سایه‌ی هر شیئ در تصویر برجسته شده است که می‌توان این برداشت را کرد که فیلتر‌های کانال چهارم کار آشکار سازی سایه‌ها را تا  حد خوبی انجام می‌دهند و همچنین در این فیلتر‌ها نقاط مرکزی بر خلاف دیگر فیلتر‌ها عبور داده نمی‌شوند

# # :قسمت ت

# ### :بهبود پارامتر‌های مدل پایه 

# In[34]:


model_1_improved = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=7, kernel_size=(5,5), activation='relu', input_shape=(32, 32, 3)),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(rate=0.25),
                    tf.keras.layers.Dense(units=10, activation='softmax')
                ])


# In[35]:


model_1_improved.compile(
                    optimizer= tf.keras.optimizers.Adam(learning_rate= pow(10,-4)),
                    loss= tf.keras.losses.CategoricalCrossentropy(),
                    metrics= [tf.keras.metrics.CategoricalAccuracy()]
                )


# ### :آموزش مدل بهبود یافته دسته بندی تصاویر 

# In[36]:


model_1_improved.fit(
                    x= training_inputs,
                    y= training_labels,
                    batch_size= 32,
                    epochs= 1000,
                    verbose= 0,
                    callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                               MyCustomCallBack()],
                    validation_split= 0.2,
                    shuffle=True
                )


# ### : ارزیابی مدل بهبود یافته دسته‌بندی تصاویر 

# In[37]:


test_result_1_improved = model_1_improved.evaluate(
    x= testing_inputs,
    y= testing_labels,
    verbose= 2
    )
print ("\n\nNew Accuracy on CIFAR-10 Test Data =", test_result_1_improved[1]*100,"%")


# #### با تغییرات ایجاد شده و همچنین کاهش مقدار نرخ یادگیری، تعداد تکرار‌ها بیشتر شده و زمان بیشتری را نسبت به قبل برای آموزش خود نیاز دارد و مقدار دقت این مدل در مقایسه با مدل قبل پیشرفت دارد

# ### :پیش‌آموزش مدل بازشناسی اعداد با مدل جدید 

# In[61]:


model_2_improved = tf.keras.Sequential(name='numbers_cnn_improved')
for layer in model_1_improved.layers[:-1]: # exclude final layer
    model_2_improved.add(layer)
model_2_improved.add(tf.keras.layers.Dense(units=10, activation='softmax', name='new_final_layer_2')) # add a new final layer
model_2_improved.summary()


# In[62]:


model_2_improved.compile(
                    optimizer= tf.keras.optimizers.Adam(learning_rate= pow(10,-4)),
                    loss= tf.keras.losses.CategoricalCrossentropy(),
                    metrics= [tf.keras.metrics.CategoricalAccuracy()]
                )


# ### :آموزش مدل بازشناسی اعداد بهبود‌یافته

# In[63]:


model_2_improved.fit(
                    x= training_inputs_2,
                    y= training_labels_2,
                    batch_size= 32,
                    epochs= 1000,
                    verbose= 0,
                    callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                               MyCustomCallBack()],
                    validation_split= 0.2,
                    shuffle=True
                )


# #### آزمایش این مدل نسبت به قبل سریع‌تر بوده و مدت زمان اجرای هر تکرار بسیار کمتر از حالت قبل است اما در مجموع زمان بیشتری نسبت به حالت قبل برای آموزش صرف می‌شود

# ### :ارزیابی مدل بازشناسی اعداد بهبود‌یافته

# In[64]:


test_result_2_improved = model_2_improved.evaluate(
    x= testing_inputs_2,
    y= testing_labels_2,
    verbose= 2
    )
print ("\n\nNew Accuracy on Numbers Test Data =", test_result_2_improved[1]*100,"%")


# #### مشاهده می‌شود که با بهبود پارامتر‌های مدل پایه، مقدار دقت مدل نهایی کمی بهبود یافته و این مدل با دقت خوبی می‌تواند سیگنال اعداد را تشخیص دهد

# # :آزمایش مدل نهایی

# In[65]:


# select a random test input from numbers:
random_index = np.random.randint(0, testing_inputs_2.shape[0]-1)
selected_wave = testing_inputs_2[random_index]
selected_wave = np.expand_dims(selected_wave, axis=0)
testing_inputs_waves[random_index].make_audio()


# In[66]:


print("output label =",np.argmax(model_2_improved.predict(selected_wave))) # TEST

