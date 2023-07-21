import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAvgPool2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, plot_model
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np
import random
import string

# 字符包含所有数字和所有大小写英文字母，一共 62 个
characters = string.digits + string.ascii_letters
# 类别数
num_classes = len(characters)
# 批次大小
batch_size = 64
# 训练集批次数
# 训练集大小相当于是 batch_size*1000=64000
train_steps = 1000
# 测试集批次数
# 测试集大小相当于是 batch_size*100=6400
test_steps = 100
# 周期数
epochs = 20
# 图片宽度
width = 160
# 图片高度
height = 60


# 自定义数据生成器
# 我们这里的验证码数据集使用 captcha 模块生产出来的，一边生产一边训练，可以认为数据集是无限的。
# Sequence是Keras中用于生成数据批次的基类，它允许我们在训练过程中使用多线程来生成数据，并且能够自动地进行并行处理。
class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=160, height=60):
        # 字符集
        self.characters = characters
        # 批次大小
        self.batch_size = batch_size
        # 生成器生成多少个批次的数据
        self.steps = steps
        # 验证码长度
        self.n_len = n_len
        # 验证码图片宽度
        self.width = width
        # 验证码图片高度
        self.height = height
        # 字符集长度
        self.num_classes = len(characters)
        # 用于产生验证码图片
        self.image = ImageCaptcha(width=self.width, height=self.height)
        # 用于保存最近一个批次验证码字符
        self.captcha_list = []

    # 获得 index 位置的批次数据
    def __getitem__(self, index):
        # 初始化数据用于保存验证码图片
        x = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        # 初始化数据用于保存标签
        # n_len 是多任务学习的任务数量，这里是 4 个任务，batch 批次大小，num_classes 分类数量
        y = np.zeros((self.n_len, self.batch_size, self.num_classes), dtype=np.uint8)
        # 数据清 0
        self.captcha_list = []
        # 生产一个批次数据
        for i in range(self.batch_size):
            # 随机产生验证码
            captcha_text = ''.join([random.choice(self.characters) for _ in range(self.n_len)])
            self.captcha_list.append(captcha_text)
            # 生成验证码图片数据并进行归一化处理
            x[i] = np.array(self.image.generate_image(captcha_text)) / 255.0
            # j -> 任务(0-3),i -> batch(0-63),ch -> 分类(单个字符)
            for j, ch in enumerate(captcha_text):
                # 设置标签，one-hot 格式
                y[j, i, self.characters.find(ch)] = 1
        # 返回一个批次的数据和标签
        return x, [y[0], y[1], y[2], y[3]]

    # 返回批次数量
    def __len__(self):
        return self.steps


# 测试生成器
# 一共一个批次，批次大小也是 1
data = CaptchaSequence(characters, batch_size=1, steps=1)
fig, axs = plt.subplots(2, 2, figsize=(10, 3))

for i in range(4):
    # 产生一个批次的数据
    x, y = data[0]
    # 在子图中显示图片
    axs[i // 2, i % 2].imshow(x[0])
    # 验证码字符和对应编号
    axs[i // 2, i % 2].set_title(data.captcha_list[0])
    axs[i // 2, i % 2].axis('off')

plt.tight_layout()
plt.show()

if not os.path.exists('Best_Captcha.h5'):
    # 载入预训练的 resnet50 模型
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))
    # 设置输入
    inputs = Input((height, width, 3))
    # 使用 resnet50 进行特征提取
    x = resnet50(inputs)
    # 平均池化
    x = GlobalAvgPool2D()(x)
    # 把验证码识别的 4 个字符看成是 4 个不同的任务,每个任务负责识别 1 个字符
    # 任务 1 识别第 1 个字符，任务 2 识别第 2 个字符，任务 3 识别第 3 个字符，任务 4 识别第4 个字符
    x0 = Dense(num_classes, activation='softmax', name='out0')(x)
    x1 = Dense(num_classes, activation='softmax', name='out1')(x)
    x2 = Dense(num_classes, activation='softmax', name='out2')(x)
    x3 = Dense(num_classes, activation='softmax', name='out3')(x)
    # 定义模型
    model = Model(inputs, [x0, x1, x2, x3])

    # 4 个任务我们可以定义 4 个 loss
    # loss_weights 可以用来设置不同任务的权重，验证码识别的 4 个任务权重都一样
    model.compile(loss={'out0': 'categorical_crossentropy',
                        'out1': 'categorical_crossentropy',
                        'out2': 'categorical_crossentropy',
                        'out3': 'categorical_crossentropy'},
                  loss_weights={'out0': 1,
                                'out1': 1,
                                'out2': 1,
                                'out3': 1},
                  optimizer=SGD(learning_rate=1e-2, momentum=0.9),
                  metrics=['acc'])
    # 监控指标统一使用 val_loss
    # 使用 EarlyStopping 来让模型停止，连续 6 个周期 val_loss 没有下降就结束训练
    # CSVLogger 保存训练数据
    # ModelCheckpoint 保存所有训练周期中 val_loss 最低的模型
    # ReduceLROnPlateau 学习率调整策略，连续 3 个周期 val_loss 没有下降当前学习率乘以0.1
    callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1),
                 CSVLogger('Captcha.csv'),
                 ModelCheckpoint('Best_Captcha.h5', monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]
    # 训练模型,若无需查看模型训练过程可去掉history
    history = model.fit(x=CaptchaSequence(characters, batch_size=batch_size, steps=train_steps),
                        epochs=epochs,
                        validation_data=CaptchaSequence(characters, batch_size=batch_size, steps=test_steps),
                        callbacks=callbacks)

# 载入训练好的模型
model = load_model('Best_Captcha.h5')
# 测试模型，随机生成验证码
# 一共一个批次，批次大小也是 1
data = CaptchaSequence(characters, batch_size=1, steps=1)
for i in range(2):
    # 产生一个批次的数据
    x, y = data[0]
    # 预测结果
    pred = model.predict(x)
    # 获得对应编号
    captcha = np.argmax(pred, axis=-1)[:, 0]
    # 根据编号获得对应验证码
    pred = ''.join([characters[x] for x in captcha])
    # 显示图片
    plt.imshow(x[0])
    # 验证码字符和对应编号
    plt.title('real:%s\npred:%s' % (data.captcha_list[0], pred))
    plt.axis('off')
    plt.show()

# 自定义四个验证码
captcha_texts = ['0oO0', '1ilj', 'xXwW', 'sSkK']

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, captcha_text in enumerate(captcha_texts):
    image = ImageCaptcha(width=160, height=60)
    # 数据归一化
    x = np.array(image.generate_image(captcha_text)) / 255.0
    # 给数据增加一个维度变成 4 维
    x = np.expand_dims(x, axis=0)
    # 预测结果
    pred = model.predict(x)
    # 获得对应编号
    captcha = np.argmax(pred, axis=-1)[:, 0]
    # 根据编号获得对应验证码
    pred = ''.join([characters[x] for x in captcha])

    # 在子图中显示图片和预测结果
    axs[i // 2, i % 2].imshow(x[0])
    axs[i // 2, i % 2].set_title('real:%s\npred:%s' % (captcha_text, pred))
    axs[i // 2, i % 2].axis('off')

plt.axis('off')
plt.tight_layout()
plt.show()


# 计算准确率，区分大小写
def accuracy(test_steps=100):
    # 用于统计准确率
    acc_sum = 0
    for x, y in CaptchaSequence(characters, batch_size=batch_size, steps=test_steps):
        # 进行一个批次的预测
        pred = model.predict(x)
        # 获得对应编号
        pred = np.argmax(pred, axis=-1)
        # 获得标签数据
        label = np.argmax(y, axis=-1)
        # 计算这个批次的准确率然后累加到总的准确率统计中
        acc_sum += (pred == label).all(axis=0).mean()
    # 返回平均准确率
    return acc_sum / test_steps


print('-----------------------------------------------------')
# 计算准确率，区分大小写
print(f'模型准确率（区分大小写）:{accuracy()}')
print('-----------------------------------------------------')


# 计算准确率，忽略大小写
def accuracy2(test_steps=100):
    # 用于统计准确率
    acc_sum = 0
    for x, y in CaptchaSequence(characters, batch_size=batch_size, steps=test_steps):
        # 进行一个批次的预测
        pred = model.predict(x)
        # 获得对应编号
        pred = np.argmax(pred, axis=-1).T
        # 保存预测值
        pred_list = []
        # 把验证码预测值转小写后保存
        for c in pred:
            # 根据编号获得对应验证码
            temp_c = ''.join([characters[x] for x in c])
            # 字母都转小写后保存
            pred_list.append(temp_c.lower())
        # 获得标签数据
        label = np.argmax(y, axis=-1).T
        # 保存标签
        label_list = []
        # # 把验证码标签值转小写后保存
        for c in label:
            # 根据编号获得对应验证码
            temp_c = ''.join([characters[x] for x in c])
            # 字母都转小写后保存
            label_list.append(temp_c.lower())
        # 计算这个批次的准确率然后累加到总的准确率统计中
        acc_sum += (np.array(pred_list) == np.array(label_list)).mean()
    # 返回平均准确率
    return acc_sum / test_steps


print('-----------------------------------------------------')
# 计算准确率，忽略大小写
print(f'模型准确率（不区分大小写）:{accuracy2()}')
print('-----------------------------------------------------')
