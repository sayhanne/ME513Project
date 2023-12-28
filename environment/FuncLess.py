
def FuncLess(RealTime, ta, dt):

    Lim1 = ta - dt*0.25

    if(RealTime < (Lim1)):

        Don = True
    else:
        Don = False


    return Don