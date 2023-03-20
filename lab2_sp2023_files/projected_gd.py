import asgn_source as asou
import numpy as np
import scipy


def project_coordinate(x,constraint_min,constraint_max,n):
    
    ### write your code here      
    
    return x_projected




def run_projected_GD(constraint_min,constraint_max, Q,b, c,n):
    
    ### write your code here 
    
    return result



if __name__ == '__main__':
    n =25
    np.random.seed()
    constraint_min,constraint_max, Q_val,b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(constraint_min,constraint_max,Q_val,b_val, c_val,n)
    


