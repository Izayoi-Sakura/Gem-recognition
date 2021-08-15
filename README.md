# Gem-recognition

# 项目简介

本项目旨在对不同宝石的图片进行学习以实现对宝石类别的识别。

# 一、项目背景

在宝石鉴别场景下，往往需要用到专业仪器抑或是从业者丰富的经验，在对高价值宝石进行鉴定时毫无疑问是合适的；但对于一般价值或对精度要求不甚高的场合，人工鉴别就显得颇为麻烦。所以，能够在线上对宝石进行类别的一般识别是非常有必要的。

# 二、数据集简介

本项目所使用的数据集包括811张图片，分为25种宝石类型并进行了标注，如下所示：

![](
https://ai-studio-static-online.cdn.bcebos.com/cbe09d0a7a1e40fca5f514814fbd08194f7f64ec57f3467d82cd712ac4bceaf2)
![](https://ai-studio-static-online.cdn.bcebos.com/9055dcd7dcbc419a9c6d676eed4b4e635c5e68773feb4024bd02ffcb6be8eb5f)

## 1.数据加载和预处理


```python
'''
# 配置需求参数
'''
train_parameters = {
    "input_size": [3, 64, 64],                           #输入图片的shape
    "class_dim": -1,                                     #分类数
    'augment_path' : '/home/aistudio/augment',           #数据增强图片目录
    "src_path":"data/data55032/archive_train.zip",       #原始数据集路径
    "target_path":"/home/aistudio/data/dataset",        #要解压的路径 
    "train_list_path": "./train_data.txt",              #train_data.txt路径
    "eval_list_path": "./val_data.txt",                  #eval_data.txt路径
    "label_dict":{},                                    #标签字典
    "readme_path": "/home/aistudio/data/readme.json",   #readme.json路径
    "num_epochs": 20,                                    #训练轮数
    "train_batch_size": 64,                             #批次的大小
    "learning_strategy": {                              #优化函数相关的配置
        "lr": 0.001                                     #超参数学习率
    } 
}
```

```python
def unzip_data(src_path,target_path):

    '''
    解压原始数据集，将src_path路径下的zip包解压至data/dataset目录下
    '''

    if(not os.path.isdir(target_path)):    
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")
```

```python
def get_data_list(target_path,train_list_path,eval_list_path, augment_path):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存的文件夹名称
    data_list_path=target_path
    class_dirs = os.listdir(data_list_path)
    if 'MACOSX' in class_dirs:
        class_dirs.remove('MACOSX')
    # #总的图像数量
    all_class_images = 0
    # #存放类别标签
    class_label=0
    # #存放类别数目
    class_dim = 0
    # #存储要写进eval.txt和train.txt中的内容
    trainer_list=[]
    eval_list=[]
    #读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            #每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            #统计每个类别有多少张图片
            class_sum = 0
            #获取类别路径 
            path = os.path.join(data_list_path,class_dir)
            # print(path)
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:                                  # 遍历文件夹下的每个图片
                if img_path =='.DS_Store':
                    continue
                name_path = os.path.join(path,img_path)                       # 每张图片的路径
                if class_sum % 15 == 0:                                 # 每10张图片取一个做验证数据
                    eval_sum += 1                                       # eval_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1 
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
                class_sum += 1                                          #每类图片的数目
                all_class_images += 1                                   #所有类图片的数目 
            # ----------------------------------数据增强----------------------------------
            aug_path = os.path.join(augment_path, class_dir)
            for img_path in os.listdir(aug_path):                                  # 遍历文件夹下的每个图片
                name_path = os.path.join(aug_path,img_path)                       # 每张图片的路径
                trainer_sum += 1 
                trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
                all_class_images += 1                                   #所有类图片的数目
            # ----------------------------------------------------------------------------
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir             #类别名称
            class_detail_list['class_label'] = class_label          #类别标签
            class_detail_list['class_eval_images'] = eval_sum       #该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
            class_detail.append(class_detail_list)  
            #初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1
            
    #初始化分类数
    train_parameters['class_dim'] = class_dim
    print(train_parameters)
    #乱序  
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image) 
    #乱序        
    random.shuffle(trainer_list) 
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image) 

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path                  #文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'],'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')   
```

```python
def data_reader(file_list):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                img = Image.open(img_path) 
                if img.mode != 'RGB': 
                    img = img.convert('RGB') 
                img = img.resize((64, 64), Image.BILINEAR)
                img = np.array(img).astype('float32') 
                img = img.transpose((2, 0, 1))  # HWC to CHW 
                img = img/255                   # 像素值归一化 
                yield img, int(lab) 
    return reader
```

```python
!pip install Augmentor
Looking in indexes: https://mirror.baidu.com/pypi/simple/
Requirement already satisfied: Augmentor in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.2.8)
Requirement already satisfied: tqdm>=4.9.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Augmentor) (4.36.1)
Requirement already satisfied: future>=0.16.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Augmentor) (0.18.0)
Requirement already satisfied: numpy>=1.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Augmentor) (1.16.4)
Requirement already satisfied: Pillow>=5.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/si
```

```python

def proc_img(src):
    for root, dirs, files in os.walk(src):
        if '__MACOSX' in root:continue
        for file in files:            
            src=os.path.join(root,file)
            img=Image.open(src)
            if img.mode != 'RGB': 
                    img = img.convert('RGB') 
                    img.save(src)            


if __name__=='__main__':
    proc_img(r"data/dataset")
```

## 2.数据增强

```python
import os, Augmentor
import shutil, glob

if not os.path.exists(augment_path): # 控制不重复增强数据
    for root, dirs, files in os.walk("data/dataset", topdown=False):
        for name in dirs:
            path_ = os.path.join(root, name)
            if '__MACOSX' in path_:continue
            print('数据增强：',os.path.join(root, name))
            print('image：',os.path.join(root, name))
            p = Augmentor.Pipeline(os.path.join(root, name),output_directory='output')
            p.rotate(probability=0.6, max_left_rotation=2, max_right_rotation=2)
            p.zoom(probability=0.6, min_factor=0.9, max_factor=1.1)
            p.random_distortion(probability=0.4, grid_height=2, grid_width=2, magnitude=1)

            count = 1000 - len(glob.glob(pathname=path_+'/*.jpg'))
            p.sample(count, multi_threaded=False)
            p.process()

    print('将生成的图片拷贝到正确的目录')
    for root, dirs, files in os.walk("data/dataset", topdown=False):
        for name in files:
            path_ = os.path.join(root, name)
            if path_.rsplit('/',3)[2] == 'output':
                type_ = path_.rsplit('/',3)[1]
                dest_dir = os.path.join(augment_path ,type_) 
                if not os.path.exists(dest_dir):os.makedirs(dest_dir) 
                dest_path_ = os.path.join(augment_path ,type_, name) 
                shutil.move(path_, dest_path_)
    print('删除所有output目录')
    for root, dirs, files in os.walk("data/dataset", topdown=False):
        for name in dirs:
            if name == 'output':
                path_ = os.path.join(root, name)
                shutil.rmtree(path_)
    print('完成数据增强')

```

```python
#每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
    
#生成数据列表   
get_data_list(target_path,train_list_path,eval_list_path,augment_path)

'''
构造数据提供器
'''
train_reader = paddle.batch(data_reader(train_list_path),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),
                            batch_size=batch_size,
                            drop_last=True)

```

```python

Batch=0
Batchs=[]
all_train_accs=[]
def draw_train_acc(Batchs, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()

all_train_loss=[]
def draw_train_loss(Batchs, train_loss):
    title="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()

```


# 三、模型选择和开发

详细说明你使用的算法。此处可细分，如下所示：

## 1.模型定义


```python
#定义网络
class MyDNN(fluid.dygraph.Layer):
    '''
    卷积神经网络
    '''
    def __init__(self):
        super(MyDNN,self).__init__()
        self.hidden1=fluid.dygraph.Linear(3*64*64,1000, act='relu')
        self.hidden2=fluid.dygraph.Linear(1000,500, act='relu')
        self.hidden3=fluid.dygraph.Linear(500,100, act='relu')
        self.out = fluid.dygraph.Linear(input_dim=100, output_dim=25, act='softmax')

    def forward(self,input):
        x = fluid.layers.reshape(input,shape=[-1,3*64*64])
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x
```

## 2.模型训练


```python

with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
    print(train_parameters['class_dim'])
    print(train_parameters['label_dict'])
    model=MyDNN() #模型实例化
    model.train() #训练模式
    opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    epochs_num=train_parameters['num_epochs'] #迭代次数
    
    for pass_num in range(epochs_num):
        for batch_id,data in enumerate(train_reader()):
            images = np.array([x[0] for x in data]).astype('float32').reshape(-1, 3,64,64)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]

            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)

            predict=model(image) #数据传入model
            
            loss=fluid.layers.cross_entropy(predict,label)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            
            acc=fluid.layers.accuracy(predict,label)#计算精度
            
            if batch_id!=0 and batch_id%5==0:
                Batch = Batch+5 
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])
                
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
            avg_loss.backward()       
            opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
            model.clear_gradients()   #model.clear_gradients()来重置梯度
    fluid.save_dygraph(model.state_dict(),'MyDNN')#保存模型

draw_train_acc(Batchs,all_train_accs)
draw_train_loss(Batchs,all_train_loss)
```


## 3.模型评估


```python
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    for batch_id,data in enumerate(eval_reader()):#测试集
        images = np.array([x[0] for x in data]).astype('float32').reshape(-1, 3,64,64)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)       
        predict=model(image)       
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)
```


## 4.模型预测


```python
import os
import zipfile

def unzip_infer_data(src_path,target_path):
    '''
    解压预测数据集
    '''
    if(not os.path.isdir(target_path)):     
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path) 
    if img.mode != 'RGB': 
        img = img.convert('RGB') 
    img = img.resize((64, 64), Image.BILINEAR)
    img = np.array(img).astype('float32') 
    img = img.transpose((2, 0, 1))  # HWC to CHW 
    img = img/255                # 像素值归一化 
    return img


infer_src_path = '/home/aistudio/data/data55032/archive_test.zip'
infer_dst_path = '/home/aistudio/data/archive_test'
unzip_infer_data(infer_src_path,infer_dst_path)
```

```python
 label_dic = train_parameters['label_dict']

'''
模型预测
'''
with fluid.dygraph.guard():
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    
    #展示预测图片
    infer_path='data/archive_test/alexandrite_3.jpg'
    img = Image.open(infer_path)
    plt.imshow(img)          #根据数组绘制图像
    plt.show()               #显示图像

    #对预测图片进行预处理
    infer_imgs = []
    infer_imgs.append(load_image(infer_path))
    infer_imgs = np.array(infer_imgs)
   
    for i in range(len(infer_imgs)):
        data = infer_imgs[i]
        dy_x_data = np.array(data).astype('float32')
        dy_x_data=dy_x_data[np.newaxis,:, : ,:]
        img = fluid.dygraph.to_variable(dy_x_data)
        out = model(img)
        lab = np.argmax(out.numpy())  #argmax():返回最大数的索引

        print("第{}个样本,被预测为：{},真实标签为：{}".format(i+1,label_dic[str(lab)],infer_path.split('/')[-1].split("_")[0]))
        
print("结束")
```


