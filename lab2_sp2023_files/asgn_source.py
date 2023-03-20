import numpy as np
import scipy.optimize as so


import numpy as np
def get_Q_b_c (n):
    Q = np . random . rand (n ,n)-0.5
    Q = 10*Q @ Q.T+np.eye(n)
    b = 5*( np . random . rand (n)-0.5 )
    c = 2*( np . random . rand (1)-0.5 )
    return Q ,b ,c
    # Example :
#Q ,b , c = get_Q_b_c ( 10 )



def opfun(x,Q,b,c):
    return 0.5*(x.transpose() @ Q @ x) +  (b.transpose()@ x) + c



def get_box_constraints(Q,b,c,width,delta,n):
    sp = np.random.rand(n)
    res=so.minimize(opfun,sp,args= (Q,b,c))
    opt_point = res['x']
    
    pos = 0 
    if np.random.rand()> 0.5 :
        pos = 1
        
    
    
    
    if pos ==1 :
        box_min = np.ceil(opt_point)  + float(delta) 
        box_max =  np.ceil(opt_point)  + float(delta) + float(width)
    else :
        box_max = np.floor(opt_point) - float(delta)
        box_min =  np.floor(opt_point) - float(delta) - float(width) 
        
    return box_min,box_max, opt_point
    
    
def get_parameters(n):
    width,delta = 5, 20
    Q ,b , c = get_Q_b_c ( n )
    constraint_min,constraint_max, opt_point = get_box_constraints(Q,b,c,width,delta,n)
    
    return constraint_min,constraint_max, Q,b,c
    
    