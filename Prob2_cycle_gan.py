#------------------------------ Import Module --------------------------------#
import numpy as np
import cv2
import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt 
import random

# 使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.device('/device:GPU:2')

#--------------------------------- Parameter ---------------------------------#
image_heigh=60            # 統一圖片高度
image_width=60            # 統一圖片寬度
data_number=10000          # 要取多少筆data來train
batch_size=5             # 多少筆data一起做訓練
epoch_num=30              # 執行多少次epoch


#---------------------------------- Function ---------------------------------#
# 讀取圖片
def read_image(path,data_number):
    imgs = os.listdir(path)      # 獲得該路徑下所有的檔案名稱
    total_image=np.zeros([data_number,image_heigh,image_width,3])   
    # 依序將每張圖片儲存進矩陣total_image當中
    for num_image in range(0,data_number):
        filePath=path+'//'+imgs[num_image]    # 圖片路徑
        cv_img=cv2.imread(filePath)  # 取得圖片
        total_image[num_image,:,:,:] = cv2.resize(cv_img, (image_heigh, image_width), interpolation=cv2.INTER_CUBIC)  # resize並且存入total_image當中      
    return total_image

# 建立convolution層
def conv(input_node,layer_node,kernel_height_width,is_activation,name="convolution_layer"):
    # input_node 為輸入的節點
    # kernel_Size為filter的大小, 其中height與width都相同
    # is_activation 決定輸出是否要過relu
    # name 為此convolution的名稱
    with tf.variable_scope(name):       
        output=tf.layers.conv2d(
                inputs=input_node,
                filters=layer_node,
                kernel_size=[kernel_height_width,kernel_height_width],
                padding="same"
                )
        if is_activation:
            output=tf.nn.relu(output)
        return output
    
# 建立Transpose Convolution層
def conv_trans(input_node,kernel_height_width,layer_node,is_activation,name="convolution_transpose_layer"):        
    # input_node 為輸入的節點
    # kernel_Size為filter的大小, 其中height與width都相同
    # output_shape為要輸出的tensor的shape
    # is_activation 決定輸出是否要過relu
    # name 為此convolution的名稱
    with tf.variable_scope(name): 
        output=tf.layers.conv2d_transpose(
            inputs=input_node,
            filters=layer_node,
            kernel_size=[kernel_height_width,kernel_height_width],
            strides=1,
            padding="same",
            )        
        if is_activation:
            output=tf.nn.relu(output)
        return output
    
# 建立ResNet block層
def ResNet(input_node,kernel_height_width,is_activation,name="resnet"):
    # input_node 為輸入的節點
    # kernel_Size為filter的大小, 其中height與width都相同
    filters_number=input_node.shape.as_list()[3]    # 取得feature map數量
    with tf.variable_scope(name):
        # ResNet第一層
        output1=tf.layers.conv2d_transpose(            
                inputs=input_node,
                filters=filters_number,
                kernel_size=[kernel_height_width,kernel_height_width],
                strides=1,
                padding="same"
                )
        if is_activation:
            output1=tf.nn.relu(output1)
        # ResNet第二層
        output2=tf.layers.conv2d_transpose(            
                inputs=input_node,
                filters=filters_number,
                kernel_size=[kernel_height_width,kernel_height_width],
                strides=1,
                padding="same"
                )
        if is_activation:
            output2=tf.nn.relu(output2)
        # 輸出加總
        output=output1+output2
        return output
            
# 建立Generator          
def Generator(input_node,name="generator"):
    # input_node 為輸入的節點
    with tf.variable_scope(name):
        # 輸入的圖片先過3層CNN
        cnn_layer1=conv(input_node,32,3,True,name="cnn_layer1")
        cnn_layer2=conv(cnn_layer1,64,3,True,name="cnn_layer2")
        cnn_layer3=conv(cnn_layer2,128,3,True,name="cnn_layer3")
        # 通過5層ResNet
        resnet_layer1=ResNet(cnn_layer3,3,True,name="resnet_layer1")
        resnet_layer2=ResNet(resnet_layer1,3,True,name="resnet_layer2")
        resnet_layer3=ResNet(resnet_layer2,3,True,name="resnet_layer3")
        resnet_layer4=ResNet(resnet_layer3,3,True,name="resnet_layer4")
        resnet_layer5=ResNet(resnet_layer4,3,True,name="resnet_layer5")
        # 輸出前通過3層CNN
        cnn_layer4=conv_trans(resnet_layer5,3,128,True,name="cnn_layer4")
        cnn_layer5=conv_trans(cnn_layer4,3,64,True,name="cnn_layer5")
        cnn_layer6=conv_trans(cnn_layer5,3,32,True,name="cnn_layer6")
        cnn_layer7=conv(cnn_layer6,3,3,False,name="cnn_layer7")
        # 輸出
        #output=tf.nn.tanh(cnn_layer7,"t1")
        output=tf.sigmoid(cnn_layer7,"t1")
        return output
    
# 建立Discriminator
def Discriminator(input_node,name="discriminator"):
    with tf.variable_scope(name):
        cnn_layer1=conv(input_node,64,3,True,name="cnn_layer1")
        cnn_layer2=conv(cnn_layer1,64*2,3,True,name="cnn_layer2")
        cnn_layer3=conv(cnn_layer2,64*4,3,True,name="cnn_layer3")
        cnn_layer4=conv(cnn_layer3,64*8,3,True,name="cnn_layer4")
        cnn_layer5=conv(cnn_layer4,1,3,False,name="cnn_layer5")
        return(cnn_layer5)
        
#---------------------------------- Cycle_GAN --------------------------------#      
class Cycle_GAN():
    # 初始化
    def initialize(self):
        self.learning_rate=5*1e-6            # 初始learning rate
        self.fake_pool_max_number=100      # fake pool最大data數量
        self.fake_pool_number=0            # fake pool目前data數量
        # 紀錄Loss
        self.loss_GA=[]
        self.loss_GB=[]
        self.loss_DA=[]
        self.loss_DB=[]
        

    # 傳入training data
    def input_data(self):
        path1=r'/home/alantao/deep learning/DL HW3/cartoon'
        path2=r'/home/alantao/deep learning/DL HW3/animation'
        self.cartoon_data=read_image(path1,data_number)
        self.animation_data=read_image(path2,data_number)
        # 修改資料型態
        #self.cartoon_data=self.cartoon_data.reshape([-1,image_heigh*image_width*3])  # 把每個顏色的2為圖片(連同RGB)拉長  
        self.cartoon_data/=255      # normalize
        #self.animation_data=self.animation_data.reshape([-1,image_heigh*image_width*3])  # 把每個顏色的2為圖片(連同RGB)拉長  
        self.animation_data/=255    # normalize
        
    # Fake pool
    def Fake_pool(self,new_data_A,new_data_B):
        add_data_number = np.size(new_data_A,0)    # 這次要加入的data數量
        # 初始化(第1次)
        if self.fake_pool_number == 0:
            self.fake_pool_A = new_data_A
            self.fake_pool_B = new_data_B
            self.fake_pool_number=add_data_number
        # 若pool已滿, 則從已存在的pool中隨機找幾個被取代
        elif self.fake_pool_number + add_data_number > self.fake_pool_max_number:
            for substitute_times in range(0,add_data_number-1):
                random_index = random.randint(0,self.fake_pool_max_number-1) #隨便挑
                self.fake_pool_A[random_index,:,:,:] = new_data_A[substitute_times,:,:,:]  # 被新資料取代
                self.fake_pool_B[random_index,:,:,:] = new_data_B[substitute_times,:,:,:]  # 被新資料取代
        # 若並非第1次且pool未滿, 則直接加入
        else:
            self.fake_pool_A = np.concatenate((self.fake_pool_A,new_data_A),axis=0)
            self.fake_pool_B = np.concatenate((self.fake_pool_B,new_data_B),axis=0)
            self.fake_pool_number+=add_data_number
            
    # 建立NN
    def Model(self):
        
        ## 建立placeholder ##
        # 放置 A domain(cartoon)的data 
        self.input_domain_A=tf.placeholder(tf.float32,[None,image_width,image_heigh,3],name="input_domain_A")
        # 放置 B domain(animation)的data 
        self.input_domain_B=tf.placeholder(tf.float32,[None,image_width,image_heigh,3],name="input_domain_B")
        # 放置 A-->B domain(fake animation)的data
        self.fake_domain_B=tf.placeholder(tf.float32,[None,image_width,image_heigh,3],name="fake_domain_B")
        # 放置 B-->A domain(fake cartoon)的data
        self.fake_domain_A=tf.placeholder(tf.float32,[None,image_width,image_heigh,3],name="fake_domain_A")
        # 學習率learning rate變數
        self.lr=tf.placeholder(tf.float32,shape=[],name="lr")
        
        
        ## 建立模型 ##
        with tf.variable_scope("Model") as scope:
            # 把self.input_domain_A(A domain)通過generator_A,得到self.fake_B(true A--> fake B)
            self.fake_B=Generator(self.input_domain_A,name="generator_A")
            # 把self.input_domain_B(B domain)通過generator_B,得到self.fake_A(true B-->fake A)
            self.fake_A=Generator(self.input_domain_B,name="generator_B")
            
            # 檢驗true A圖片通過discriminator_A,得到分數self.check_true_A
            self.check_true_A=Discriminator(self.input_domain_A,name="discriminator_A")
            # 檢驗true B圖片通過discriminator_B,得到分數self.check_true_B
            self.check_true_B=Discriminator(self.input_domain_B,name="discriminator_B")
            
            scope.reuse_variables()   # 共享變數名稱
            
            # 檢驗fake A圖片通過discriminator_A,得到分數self.check_fake_A
            self.check_fake_A=Discriminator(self.fake_A,name="discriminator_A")
            # 檢驗fake B圖片通過discriminator_B,得到分數self.check_fake_B
            self.check_fake_B=Discriminator(self.fake_B,name="discriminator_B")
            
            # 把self.fake_B(true A--> fake B)通過generator_B,得到self.cycle_A
            self.cycle_A=Generator(self.fake_B,name="generator_B")
            # 把self.fake_A(true B--> fake A)通過generator_A,得到self.cycle_B
            self.cycle_B=Generator(self.fake_A,name="generator_A")
            
            scope.reuse_variables()   # 共享變數名稱
            
            # 取出self.fake_domain_A(歷史的self.fake_A)通過discriminator_A,得到分數self.check_pool_fake_A
            self.check_pool_fake_A=Discriminator(self.fake_domain_A,name="discriminator_A")
            # 取出self.fake_domain_B(歷史的self.fake_B)通過discriminator_B,得到分數self.check_pool_fake_B
            self.check_pool_fake_B=Discriminator(self.fake_domain_B,name="discriminator_B")
            
            
    # 計算loss function 
    def Loss_func(self):
        
        ## 計算Generator Loss ##
        
        # 計算Cycle Loss #
        # cycle A Loss: trueA->fakeB->cycleA得trueA與cycleA的誤差
        # cycle B Loss: trueB->fakeA->cycleB得trueB與cycleB的誤差
        cycle_loss=tf.reduce_mean(tf.abs(self.input_domain_A-self.cycle_A))\
        +tf.reduce_mean(tf.abs(self.input_domain_B-self.cycle_B))
        
        # 計算Generator造成的Discriminator Loss #
        # 使用Generator產生的fake data應該要讓Discriminator判斷成'1', 與'1'差越多表示Loss越大
        generator_DiscriminatorA_loss=tf.reduce_mean(tf.squared_difference(self.check_fake_A,1))
        generator_DiscriminatorB_loss=tf.reduce_mean(tf.squared_difference(self.check_fake_B,1))
        
        # 計算Generator的total loss
        self.generator_A_loss=cycle_loss*10+generator_DiscriminatorB_loss
        self.generator_B_loss=cycle_loss*10+generator_DiscriminatorA_loss
        
        ## 計算Discriminator Loss ##
        self.discriminator_A_loss=0.5*tf.reduce_mean(tf.square(self.check_pool_fake_A))\
        +0.5*tf.reduce_mean(tf.squared_difference(self.check_true_A,1))
        self.discriminator_B_loss=0.5*tf.reduce_mean(tf.square(self.check_pool_fake_B))\
        +0.5*tf.reduce_mean(tf.squared_difference(self.check_true_B,1))
        
        ## 使用Loss更新變數
        optimize_method=tf.train.AdamOptimizer(self.learning_rate)  # 最佳化方式
        self.model_variables=tf.trainable_variables()   # 取得所有可以做訓練的變數
        
        # 訓練Generator_A的Loss
        generator_A_variables=[var for var in self.model_variables if 'generator_A' in var.name]
        self.generator_A_trainer=optimize_method.minimize(self.generator_A_loss,var_list=generator_A_variables)
        # 訓練Generator_B的Loss
        generator_B_variables=[var for var in self.model_variables if 'generator_B' in var.name]
        self.generator_B_trainer=optimize_method.minimize(self.generator_B_loss,var_list=generator_B_variables)
        # 訓練Discriminator_A的Loss
        discriminator_A_variables=[var for var in self.model_variables if 'discriminator_A' in var.name]
        self.discriminator_A_trainer=optimize_method.minimize(self.discriminator_A_loss,var_list=discriminator_A_variables)
        # 訓練Discriminator_B的Loss
        discriminator_B_variables=[var for var in self.model_variables if 'discriminator_B' in var.name]
        self.discriminator_B_trainer=optimize_method.minimize(self.discriminator_B_loss,var_list=discriminator_B_variables)

    # 開始訓練
    def Train(self):
        self.initialize()             # 初始化
        self.input_data()             # 讀取data
        self.Model()                  # 建立模型
        self.Loss_func()              # 建立Loss
        
        # 建立Session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())      # 激活所有變數
            for epoch_times in range(0,epoch_num):           # 要執行多次epoch
                print('epoch times=',epoch_times)
                for batch_times in range(0,int(data_number/batch_size)):  # 全部的資料可以分成多少個batch
                    # 取得batch_size筆cartoon的data
                    get_A=self.cartoon_data[batch_times*batch_size:(batch_times+1)*batch_size,:]
                    # 取得batch_size筆animation的data
                    get_B=self.animation_data[batch_times*batch_size:(batch_times+1)*batch_size,:] 
                    # 取得self.fake_A(B->A)
                    temp_fake_A=self.fake_A.eval(feed_dict={self.input_domain_B:get_B})
                    # 取得self.fake_B(A->B)
                    temp_fake_B=self.fake_B.eval(feed_dict={self.input_domain_A:get_A})
                    # self.fake_A與temp_fake_B分別丟入self.fake_pool_A與self.fake_pool_B
                    self.Fake_pool(temp_fake_A,temp_fake_B)
                    
                    # 訓練Generator_A
                    self.generator_A_trainer.run(feed_dict={self.input_domain_A:get_A,
                                                            self.input_domain_B:get_B,
                                                            self.fake_domain_A:self.fake_pool_A,
                                                            self.fake_domain_B:self.fake_pool_B,                                                            
                                                            self.lr:self.learning_rate})
                    # 訓練Discriminator_B
                    self.discriminator_B_trainer.run(feed_dict={self.input_domain_A:get_A,
                                                            self.input_domain_B:get_B,
                                                            self.fake_domain_A:self.fake_pool_A,
                                                            self.fake_domain_B:self.fake_pool_B,                                                            
                                                            self.lr:self.learning_rate})
                    # 訓練Generator_B
                    self.generator_B_trainer.run(feed_dict={self.input_domain_A:get_A,
                                                            self.input_domain_B:get_B,
                                                            self.fake_domain_A:self.fake_pool_A,
                                                            self.fake_domain_B:self.fake_pool_B,                                                            
                                                            self.lr:self.learning_rate})
                    # 訓練Discriminator_A
                    self.discriminator_A_trainer.run(feed_dict={self.input_domain_A:get_A,
                                                            self.input_domain_B:get_B,
                                                            self.fake_domain_A:self.fake_pool_A,
                                                            self.fake_domain_B:self.fake_pool_B,                                                            
                                                            self.lr:self.learning_rate})
                    # 每個batch觀察Loss
                    GA=self.generator_A_loss.eval(feed_dict={self.input_domain_A:get_A,
                                                        self.input_domain_B:get_B,
                                                        self.fake_domain_A:self.fake_pool_A,
                                                        self.fake_domain_B:self.fake_pool_B,                                                            
                                                        self.lr:self.learning_rate})
                    GB=self.generator_B_loss.eval(feed_dict={self.input_domain_A:get_A,
                                                        self.input_domain_B:get_B,
                                                        self.fake_domain_A:self.fake_pool_A,
                                                        self.fake_domain_B:self.fake_pool_B,                                                            
                                                        self.lr:self.learning_rate})    
                    DA=self.discriminator_A_loss.eval(feed_dict={self.input_domain_A:get_A,
                                                        self.input_domain_B:get_B,
                                                        self.fake_domain_A:self.fake_pool_A,
                                                        self.fake_domain_B:self.fake_pool_B,                                                            
                                                        self.lr:self.learning_rate}) 
                    DB=self.discriminator_B_loss.eval(feed_dict={self.input_domain_A:get_A,
                                                        self.input_domain_B:get_B,
                                                        self.fake_domain_A:self.fake_pool_A,
                                                        self.fake_domain_B:self.fake_pool_B,                                                            
                                                        self.lr:self.learning_rate})  
                    # 紀錄每次epoch的loss
                    self.loss_GA.append(GA)
                    self.loss_GB.append(GB)
                    self.loss_DA.append(DA)
                    self.loss_DB.append(DB)
                
                # print目前狀態
                print('GA=',GA,',GB=',GB,',DA=',DA,',DB=',DB)
                print('Learning rate=',self.learning_rate)
                # 修正learning rate
                self.learning_rate*=0.8
                # Shuffle
                np.random.shuffle(self.cartoon_data)       # shuffle
                np.random.shuffle(self.animation_data)     # shuffle
            self.Test(sess)
                
    # 做測試
    def Test(self,sess):
        #self.input_data()             # 讀取data
        #self.Model()                  # 建立模型
        
        # 在plot之前要加上這行    
        %matplotlib inline      
        
        # 印出learning curve
        plt.figure(1)
        plt.plot(self.loss_GA)
        plt.xlabel('Number of batch')
        plt.ylabel('Generator A loss')
        plt.show()   
        
        plt.figure(2)
        plt.plot(self.loss_GB)
        plt.xlabel('Number of batch')
        plt.ylabel('Generator B loss')
        plt.show() 
        
        plt.figure(3)
        plt.plot(self.loss_DA)
        plt.xlabel('Number of batch')
        plt.ylabel('Discriminator A loss')
        plt.show() 
        
        plt.figure(4)
        plt.plot(self.loss_DB)
        plt.xlabel('Number of batch')
        plt.ylabel('Discriminator B loss')
        plt.show() 
        
        
        for image_num in range(0,15):  # 取3個圖片做比較
            # 原A domain圖片
            tempA=self.cartoon_data[image_num,:]
            tempA=tempA.reshape([1,image_heigh,image_width,3])
            testA=tempA.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testA)
            testA = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testA) 
            plt.show()

            # 原A domain圖片經過generatorA變成fake B domain
            temp_fake_B=self.fake_B.eval(feed_dict={self.input_domain_A:tempA})
            testB=temp_fake_B.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testB)
            testB = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testB) 
            plt.show()
            
            # 原A domain圖片經過generatorA變成fake B domain, 在經過generatorA變回cycle A domain
            temp_cycleA=self.cycle_A.eval(feed_dict={self.fake_B:temp_fake_B})
            testA2=temp_cycleA.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testA2)
            testA2 = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testA2) 
            plt.show()
            
            # 原B domain圖片
            tempB=self.animation_data[image_num,:]
            tempB=tempB.reshape([1,image_heigh,image_width,3])
            testB=tempB.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testB)
            testB = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testB) 
            plt.show()

            # 原B domain圖片經過generatorB變成fake A domain
            temp_fake_A=self.fake_A.eval(feed_dict={self.input_domain_B:tempB})            
            testA=temp_fake_A.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testA)
            testA = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testA) 
            plt.show()
            
            # 原A domain圖片經過generatorA變成fake B domain, 在經過generatorA變回cycle A domain
            temp_cycleB=self.cycle_B.eval(feed_dict={self.fake_A:temp_fake_A})
            testB2=temp_cycleB.reshape([image_heigh,image_width,3])
            b,g,r = cv2.split(testB2)
            testB2 = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
            plt.imshow(testB2) 
            plt.show()
            
            
            
            
                           
#------------------------------------ Main -----------------------------------# 
def main():
    prob2=Cycle_GAN()             # 建立Cycle_GAN()類別的物件
    prob2.Train()                 # 開始訓練
    
# 避免執行到被引用進的程式(此處沒差~)
if __name__=='__main__':
    main()