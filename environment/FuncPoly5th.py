from FuncInterval import *
from FuncLess import *

def FuncPoly5th(RealTime,  tstart,  te,  z01,  v01,  a01,  z02,  v02,  a02, dt):

    twidth = te - tstart
    Rt = (RealTime-tstart)

    tw2 = twidth*twidth
    tw3 = tw2*twidth
    tw4 = tw2*tw2
    tw5 = tw3*tw2


    p1 = -0.5*(12*z01-12*z02+6*v01*twidth+6*v02*twidth+a01*tw2-a02*tw2)/tw5
    p2 = 0.5*(30*z01-30*z02+16*v01*twidth+14*v02*twidth+3*a01*tw2-2*a02*tw2)/tw4
    p3 = -0.5*(20*z01-20*z02+12*v01*twidth+8*v02*twidth+3*a01*tw2-a02*tw2)/tw3
    p4 = 0.5*a01
    p5 = v01
    p6 = z01

    if(FuncInterval(RealTime, tstart, te, dt) == True):

        Pos = p6 + Rt*(p5 + Rt*(p4 + Rt*(p3 + Rt*(p2 + Rt*p1))))
        Vel = p5 + Rt*(2*p4 + Rt*(3*p3 + Rt*(4*p2 + 5*Rt*p1)))
        Acc = (2*p4 + Rt*(6*p3 + Rt*(12*p2 + 20*Rt*p1)))


    elif(FuncLess(RealTime, tstart, dt) == True):

          Pos = z01
          Vel = v01
          Acc = a01
    else:

        Pos = z02
        Vel = v02
        Acc = a02


    return Pos,Vel,Acc