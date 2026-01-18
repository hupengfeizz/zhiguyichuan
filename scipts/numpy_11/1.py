import numpy as np
import matplotlib.pyplot as plt


def grad(p, w):
    delt = 2*(w-20)
    w_new = w-p*delt
    return [w_new, delt]


def loss(w):
    l = (w-20)**2+5
    return l


if __name__ == '__main__':
    p = 0.1
    w0 = 2
    w = w0
    iter = 0
    min = 0.001
    x=[]
    y=[]
    while True:
        l_old = loss(w0)
        w_new, delt = grad(p, w)
        w = w_new
        l_new = loss(w)
        
        print(f'w:{w}')
        print(f'loss:{l_new}')
        y.append(l_new)
        
        iter += 1
        x.append(iter)
        print(f'iter:{iter}')
        
        if abs(l_new-l_old) < min:
            # print(f'w:{w}')
            break

    # for i in range(20):
    #     l_old = loss(w0)
    #     w_new, delt = grad(p, w)
    #     w = w_new
    #     l_new = loss(w)
        
    #     print(f'w:{w}')
    #     print(f'loss:{l_new}')
    #     y.append(l_new)
        
    #     iter += 1
    #     x.append(iter)
    #     print(f'iter:{iter}')
        
    #     if abs(l_new-l_old) < min:
    #         # print(f'w:{w}')
    #         break
    plt.figure(figsize=(10,6))
    plt.plot(x,y)
    plt.show()
    
        
        
