import matplotlib.pyplot as plt

w0=0
w1=0
mt0=0
mt1=0
xt0=0
xt1=0
vt0=0
vt1=0
p=0.95
# lr=0.02
lr=1
e=10**-6
alpha=10**-6
beta=10**-6
weight=0.9
w0_list=[]
w1_list=[]
loss_list=[]
iter_list=[]
for i in range(1,3000):
    loss0 = 1 / 10 *((63.6 - 60*w1 - w0) ** 2
                   + (65.2 - 62*w1 - w0) ** 2
                   + (66 - 64*w1 - w0) ** 2
                   + (65.5-65*w1 - w0) ** 2
                   + (66.9-66*w1 - w0) ** 2
                   + (67.1-67*w1 - w0) ** 2
                   + (67.4-68*w1 - w0) ** 2
                   + (68.3-70*w1 - w0) ** 2
                   + (70.1-72*w1 - w0) ** 2
                   + (70-74*w1 - w0) ** 2)
    #计算梯度
    Gradient0=1/10*(-2*(63.6-60*w1 - w0)-2*(65.2 - 62*w1 - w0)-2*(66 - 64*w1 - w0)-2*(65.5-65*w1 - w0)
                    -2*(66.9-66*w1 - w0)-2*(67.1-67*w1 - w0)-2*(67.4-68*w1 - w0)-2*(68.3-70*w1 - w0)
                    -2*(70.1-72*w1 - w0)-2*(70-74*w1 - w0))
    Gradient1=1/10*(-60*(63.6-60*w1 - w0)-62*(65.2 - 62*w1 - w0)-64*(66 - 64*w1 - w0)-65*(65.5-65*w1 - w0)
                    -66*(66.9-66*w1 - w0)-67*(67.1-67*w1 - w0)-68*(67.4-68*w1 - w0)-70*(68.3-70*w1 - w0)
                    -72*(70.1-72*w1 - w0)-74*(70-74*w1 - w0))
    #计算mt
    mt0=p*mt0+(1-p)*Gradient0
    mt1=p*mt1+(1-p)*Gradient1
    #计算vt
    vt0=p*vt0+(1-p)*Gradient0**2
    vt1=p*vt1+(1-p)*Gradient1**2
    #计算单位mt
    hat_mt0=mt0/(1-p**i)
    hat_mt1=mt1/(1-p**i)
    #计算单位vt
    hat_vt0=vt0/(1-p**i)
    hat_vt1=vt1/(1-p**i)

    #迭代
    w0=w0-lr*hat_mt0/(hat_vt0+e)**0.5
    w1=w1-lr*hat_mt1/(hat_vt1+e)**0.5
    #loss值
    loss1 = 1 / 10 *((63.6 - 60*w1 - w0) ** 2
                   + (65.2 - 62*w1 - w0) ** 2
                   + (66 - 64*w1 - w0) ** 2
                   + (65.5-65*w1 - w0) ** 2
                   + (66.9-66*w1 - w0) ** 2
                   + (67.1-67*w1 - w0) ** 2
                   + (67.4-68*w1 - w0) ** 2
                   + (68.3-70*w1 - w0) ** 2
                   + (70.1-72*w1 - w0) ** 2
                   + (70-74*w1 - w0) ** 2)
    w0_list.append(w0)
    w1_list.append(w1)
    loss_list.append(loss1)
    iter_list.append(i)
    #判断
    if w0<alpha and w1<alpha:
        print(f'第{i}次迭代,学习率{lr},权重{p},w0_{i}={w0},w1_{i}={w1},loss={loss1}')
        break
    if abs(loss0-loss1)<beta:
        print(f'第{i}次迭代,学习率{lr},权重{p},w0_{i}={w0},w1_{i}={w1},loss={loss1}')
        break
    print(f'第{i}次迭代,学习率{lr},权重{p},w0_{i}={w0},w1_{i}={w1},loss0={loss0},loss1={loss1}')


plt.plot(iter_list,loss_list,color='red',linestyle='--',linewidth=2,marker='o',markersize=5,markerfacecolor='blue')
plt.title('loss')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.gray()
plt.grid(True)
plt.show()