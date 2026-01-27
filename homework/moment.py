import matplotlib.pyplot as plt

w0=0
w1=0
p=0.0005
weigth_r=0.5
v0=0
v1=0
alpha=10**-6
beta=10**-6
w0_list=[]
w1_list=[]
loss_list=[]
iter_list=[]
for i in range(1,50):
    loss0 = 1 / 4 * ((63 - 60 * w1 - w0) ** 2 + (65.2 - 62 * w1 - w0) ** 2)
    #计算梯度
    Gradient0=-64.1+61*w1+w0
    Gradient1=-3911.2+3722*w1+61*w0
    #动量
    v0=weigth_r*v0+p*Gradient0
    v1=weigth_r*v1+p*Gradient1
    #迭代
    w0=w0-v0
    w1=w1-v1
    #loss值
    loss1 = 1/4*((63-60*w1-w0)**2+(65.2-62*w1-w0)**2)
    w0_list.append(w0)
    w1_list.append(w1)
    loss_list.append(loss1)
    iter_list.append(i)
    #判断
    if w0<alpha and w1<alpha:
        print(f'第{i}次迭代,学习率{p},w0_{i}={w0},w1_{i}={w1},loss={loss1}')
        break
    if abs(loss0-loss1)<beta:
        print(f'第{i}次迭代,学习率{p},w0_{i}={w0},w1_{i}={w1},loss={loss1}')
        break
    print(f'第{i}次迭代,学习率{p},动量{v0},动量{v1},w0_{i}={w0},w1_{i}={w1},loss0={loss0},loss1={loss1}')


plt.plot(iter_list,loss_list,color='red',linestyle='--',linewidth=2,marker='o',markersize=5,markerfacecolor='blue')
plt.title('loss')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.gray()
plt.grid(True)
plt.show()