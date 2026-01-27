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
    gradient_threshold = 1e-2
    x = []
    y = []
    loss_current = loss(w)
    while True:
        w_new, delt = grad(p, w)
        w = w_new
        l_new = loss(w)
        print(f'delt:{delt}')
        if abs(delt) < gradient_threshold:
            break
        else:
            print(f'w:{w},l_new:{l_new},l_new-loss_current:{l_new-loss_current}')
            y.append(l_new)

            iter += 1
            x.append(iter)
            print(f'iter:{iter}')

            if abs(l_new-loss_current) < min:
                # print(f'w:{w}')
                break

            loss_current = l_new

    # for i in range(20):
    #     w_new, delt = grad(p, w)
    #     w = w_new
    #     l_new = loss(w)
    #     # 检查是否满足条件
    #     if abs(delt) < gradient_threshold:
    #         break
    #     else:
    #         print(f'w:{w}')
    #         print(f'l_new:{l_new}')
    #         print(f'l_new-loss_current:{l_new-loss_current}')
    #         y.append(l_new)
    
    #         iter += 1
    #         x.append(iter)
    #         print(f'iter:{iter}')

    #         if abs(l_new-loss_current) < min:
    #             # print(f'w:{w}')
    #             break

    #         loss_current = l_new

    # plt.figure(figsize=(10,6))
    plt.plot(x, y)
    plt.xlabel('iter')
    plt.ylabel('loss') 
    plt.show()
