
def FuncInterval(RealTime, ta, ts, dt):

    Lim1 = ta - dt*0.25
    Lim2 = ts - dt*0.25


    if((RealTime > Lim1) and (RealTime < Lim2)):

        Don = True
    else:

        Don = False

    return Don