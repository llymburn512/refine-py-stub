from cyipopt import minimize_ipopt
import cyipopt

import zonopy
import torch
import math

'''
Contains a zonopy only implementation of the REFINE planner with minimal implementations of the zonotope and halfspace classes.
'''

class Simple2dObstacle:
    def __init__(self, x0, y0, h0, vel, length, width):
        assert length >= 0, "The length of the obstacle is negative."
        assert width >= 0, "The width of the obstacle is negative."
        assert vel >= 0, "The velocity of the obstacle is negative."
        self.x0 = x0
        self.y0 = y0
        self.h0 = h0
        self.vel = vel
        self.length = length
        self.width = width
    
    def get_xy_zonotope_over_time_interval(self, t0, t1):
        assert t0 <= t1, "The time interval is not increasing."
        delta_t = t1 - t0
        t_midpoint = (t0 + 0.5 * delta_t)
        delta_len = self.vel * delta_t
        delta_cen_absolute = self.vel * t_midpoint
        x = self.x0 + (delta_cen_absolute * math.cos(self.h0))
        y = self.y0 + (delta_cen_absolute * math.sin(self.h0))
        G_unrot = torch.tensor([
            [(self.length + delta_len) / 2, 0],
            [0,                             self.width / 2],
        ])
        # Rotate G by h0
        R_h0 = torch.tensor([
            [math.cos(self.h0), -math.sin(self.h0)],
            [math.sin(self.h0), math.cos(self.h0)],
        ])
        G_rot = R_h0 @ G_unrot
        return zonopy.zonotope(torch.hstack([torch.tensor([[x], [y]]), G_rot]).t())
    
    def offset_by(self, delta):
        assert len(delta.shape) == 2, "The offset is not a 2D vector."
        return Simple2dObstacle(self.x0 + delta[0,0], self.y0 + delta[1,0], self.h0, self.vel, self.length, self.width)
    
    def rotate_by_radians(self, delta_h):
        h0 = self.h0 + delta_h
        rotmat = torch.tensor([
            [math.cos(delta_h), -math.sin(delta_h)],
            [math.sin(delta_h), math.cos(delta_h)],
        ])
        xy_new = rotmat @ torch.tensor([[self.x0], [self.y0]])
        x0 = xy_new[0, 0]
        y0 = xy_new[1, 0]
        return Simple2dObstacle(x0, y0, h0, self.vel, self.length, self.width)

def rotate_by_radians(zono, delta_h, x_dim=0, y_dim=1, h_dim=2):
    rotmat = torch.tensor([
        [math.cos(delta_h), -math.sin(delta_h)],
        [math.sin(delta_h), math.cos(delta_h)],
    ])
    z_mat = zono.Z.t()
    z_mat[[x_dim, y_dim], :] = rotmat @ z_mat[[x_dim, y_dim], :]
    z_mat[h_dim, 0] = z_mat[h_dim, 0] + delta_h
    return zonopy.zonotope(z_mat.t())

class Simple2dObstacleSet:
    def __init__(self, obs_list):
        self.obs_list = obs_list

    def num_obstacles(self):
        return len(self.obs_list)
    
    def obstacle(self, idx):
        return self.obs_list[idx]
    
    def get_xy_zonotopes_over_time_interval(self, t0, t1):
        return [obs.get_xy_zonotope_over_time_interval(t0, t1) for obs in self.obs_list]

    def apply_fun_over_time_interval(self, fun, t0, t1):
        return [fun(obs, t0, t1) for obs in self.obs_list]
    
    def apply_fun_to_zonotopes_over_time_interval(self, fun, t0, t1):
        return [fun(obs.get_xy_zonotope_over_time_interval(t0, t1)) for obs in self.obs_list]
    
    def offset_by(self, delta_x, delta_y):
        return Simple2dObstacleSet([obs.offset_by(delta_x, delta_y) for obs in self.obs_list])
    
    def rotate_by_radians(self, delta_h):
        return Simple2dObstacleSet([obs.rotate_by_radians(delta_h) for obs in self.obs_list])
    

class Halfspace:
    def __init__(self, a_mat, b_mat):
        if len(b_mat.shape) == 1:
            b_mat = b_mat.reshape(-1, 1)
        assert len(a_mat.shape) == 2, "The shape of the A matrix is not correct."
        assert len(b_mat.shape) == 2, "The shape of the B matrix is not correct."

        self.a_mat = a_mat
        self.b_mat = b_mat
    
    def __repr__(self) -> str:
        return f'Halfspace(a_mat={self.a_mat}, b_mat={self.b_mat})'
    
    def __str__(self) -> str:
        return self.__repr__()

    def parameterize(self, slc_G, slc_c, slc_g):
        """
        Converts the constraints from the form (PA, Pb) to the form (PA', Pb') where the 
        constraints are parameterized over the given sliceable generators.

        Assuming that pt is formed by some slicing, we have pt = sum_i ((p_i - slc_c_i) / slc_g_i) * slc_G_i
        where p = [p_1, ..., p_n]. So, we can rewrite the constraints as

        min(-(PA' @ p - Pb')) rather than min(-(PA @ pt - Pb))

        Note: if you wish to parameterize over the lambda values for a slice generator instead, you can do so by setting
            slc_G = G (of shape (d, n_p)
            slc_c = 0 (of shape (1, n_p))
            slc_g = 1 (of shape (1, n_p))
        Let P = [lambda_i, ...] of shape (n_p, 1)
        pt = ( (p_i - c_i) ./ (g_i) * G_i )
           = G @ L
        CONS = min(PA @ G @ L - Pb) <= 0 => lambda values are in collision
        """
        assert len(slc_G.shape) == 2, "The shape of slc_G is not correct."
        if len(slc_g.shape) == 1:
            slc_g = slc_g.reshape(1, -1)
        if len(slc_c.shape) == 1:
            slc_c = slc_c.reshape(1, -1)
        assert slc_c.shape[0] == 1, "The shape of slc_c is not correct."
        assert slc_g.shape[0] == 1, "The shape of slc_g is not correct."
        assert slc_G.shape[0] == self.a_mat.shape[1], "The shape of slc_G is not correct."
        assert slc_G.shape[1] == slc_c.shape[1] and slc_G.shape[1] == slc_g.shape[1], "The shapes of slc_G, slc_c, and slc_g are inconsistent."

        H = slc_G / slc_g
        a_mat_new = self.a_mat @ H
        O = torch.sum((-slc_c / slc_g) * slc_G, axis=1, keepdims=True)
        b_mat_new = self.b_mat - (self.a_mat @ O)
        return Halfspace(a_mat_new, b_mat_new)
    
    def evaluate(self, pt):
        if not hasattr(pt, '__len__'):
            pt = torch.tensor([[pt]])
        if len(pt.shape) == 1:
            pt = pt.reshape(-1, 1)
        return torch.min(-(self.a_mat @ pt - self.b_mat))
    
    def evaluate_sub_gradient(self, pt):
        if not hasattr(pt, '__len__'):
            pt = torch.tensor([[pt]])
        if len(pt.shape) == 1:
            pt = pt.reshape(-1, 1)
        return -(self.a_mat[torch.argmin(-(self.a_mat @ pt - self.b_mat)), :].item())
    
    def simplify(self, control_parameter_bounds):
        '''
        control_parameter_bounds: torch.tensor of shape (n_p, 2), the lower and upper bounds of the control parameters

        We have that min(-(A @ p - b)) <= 0 => safe so, letting p_ivl = [p_min, p_max], 
        we can remove any rows of A that satisfy the interval arithmetic expression
        -(A_i @ p_ivl - b_i) > 0 for all p_ivl in control_parameter_bounds
        which, due to the linearity of A, is equivalent to
        ( -(A_i @ p_min - b_i) > 0 ) AND ( -(A_i @ p_max - b_i) > 0 )
        <=>
        min ( -(A_i @ p_min - b_i),  -(A_i @ p_max - b_i) ) > 0

        where A_i is the ith row of A

        A has shape (m, n_p) and control_parameter_bounds has shape (n_p, 2)
        '''
        assert len(control_parameter_bounds.shape) == 2, "The shape of the control parameter bounds is not correct."
        assert control_parameter_bounds.shape[1] == 2, "The shape of the control parameter bounds is not correct."
        assert len(self.a_mat.shape) == 2, "The shape of the A matrix is not correct."
        assert self.a_mat.shape[1] == control_parameter_bounds.shape[0], "The shapes of the A matrix and the control parameter bounds are inconsistent."
        assert self.b_mat.shape[0] == self.b_mat.shape[0], "The shapes of the A matrix and the B matrix are inconsistent."

        non_trivial_rows = torch.min(-(self.a_mat @ control_parameter_bounds - self.b_mat), axis=1).values <= 0
        a_mat_new = self.a_mat[non_trivial_rows, :]
        b_mat_new = self.b_mat[non_trivial_rows, :]
        return Halfspace(a_mat_new, b_mat_new)

def find_sliceable_generator_idx(zono,slice_dim):
    '''
    zono: <zonotope>
    slice_dim: <torch.Tensor> or <list> or <int>
    , shape  []
    slice_pt: <torch.Tensor> or <list> or <float> or <int>
    , shape  []
    return <zonotope>
    '''
    if isinstance(slice_dim, list):
        slice_dim = torch.tensor(slice_dim,dtype=torch.long,device=zono.device)
    elif isinstance(slice_dim, int) or (isinstance(slice_dim, torch.Tensor) and len(slice_dim.shape)==0):
        slice_dim = torch.tensor([slice_dim],dtype=torch.long,device=zono.device)

    assert isinstance(slice_dim, torch.Tensor), 'Invalid type of input'
    assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'

    non_zero_idx = zono.generators[:,slice_dim] != 0
    assert torch.all(torch.sum(non_zero_idx,0)==1), 'There should be one generator for each slice index.'
    slice_idx = non_zero_idx.T.nonzero()[:,1]
    return slice_idx

class RefineOptProblem(cyipopt.Problem):
    def __init__(self, 
                 target_center,
                 target_generators,
                 waypoint, 
                 halfspace_constraints_over_lambda):
        '''
        minimize (target_center + target_generators @ lambdas - waypoint)^2
        subject to all halfspace_constraint.eval(lambda) <= 0
        '''
        assert target_center.shape[0] == target_generators.shape[0], "The dimensions of the target center and generators do not match."
        assert target_center.shape[0] == waypoint.shape[0], "The dimensions of the waypoint and target center do not match."

        assert target_center.ndim == 2, "The target center is not a 2D vector."
        assert target_center.shape[1] == 1, "The target center is not a column vector."

        assert waypoint.ndim == 2, "The target center is not a 2D vector."
        assert waypoint.shape[1] == 1, "The target center is not a column vector."

        self.target_center = target_center
        self.target_generators = target_generators
        self.waypoint = waypoint
        self.dim_weights = torch.ones_like(target_center)
        self.num_control_params = target_generators.shape[1]
        self.halfspace_constraints_over_lambda = halfspace_constraints_over_lambda
        self.num_constraints = len(halfspace_constraints_over_lambda)

        super().__init__(
            n=self.num_control_params,
            m=self.num_constraints,
            lb=torch.full((self.num_control_params,), -1),
            ub=torch.ones((self.num_control_params,)),
            cl=torch.full((self.num_constraints,), -1.0e19),
            cu=torch.zeros((self.num_constraints,)),
        )

    def __delta_to_wp(self, slice_lambdas):
        assert slice_lambdas.ndim == 2, "The lambdas are not a 2D vector."
        assert slice_lambdas.shape[0] == self.num_control_params, "The number of control parameters does not match the number of lambdas."
        assert slice_lambdas.shape[1] == 1, "The lambdas are not a column vector."
        return (self.target_center + self.target_generators @ slice_lambdas) - self.waypoint

    def objective(self, slice_lambdas):
        slice_lambdas = slice_lambdas.reshape(-1, 1)
        delta_to_wp = self.__delta_to_wp(slice_lambdas)
        return torch.sum(self.dim_weights * torch.abs(delta_to_wp))
    
    def gradient(self, slice_lambdas):
        # Gradient of the objective function
        #
        # d/dx |k*x + a| = k * (a + k * x) / |a + k * x|
        # for each lambda j and dim d we want to compute
        # d/(d l_j) abs(delta_to_wp_d)
        # = d/(d l_j) abs(t_c_d + sum_i(tg_d * l_i) - w_d)
        # = d/(d l_j) abs(t_c_d + sum_{i \neq j}(tg_i_d * l_i) - w_d + tg_j_d * l_j)
        # let a = t_c_d + sum_{i \neq j}(tg_i_d * l_i) - w_d
        #       = delta_to_wp_d - tg_j_d * l_j
        # = d/(d l_j) abs(a + tg_j_d * l_j)
        # = tg_j_d * (a + tg_j_d * l_j) / |a + tg_j_d * l_j|
        # = tg_j_d * ((delta_to_wp_d - tg_j_d * l_j) + tg_j_d * l_j) / |(delta_to_wp_d - tg_j_d * l_j) + tg_j_d * l_j|
        # = tg_j_d * (delta_to_wp_d) / |delta_to_wp_d|
        # = tg_j_d * sign(delta_to_wp_d)
        #
        # Then we just sum this over all dimensions, for each lambda
        #
        # NOTE: this actually returns a subgradient of the objective
        slice_lambdas = slice_lambdas.reshape(-1, 1)
        
        delta_to_wp = self.__delta_to_wp(slice_lambdas)
        grad_calc = torch.sum((self.dim_weights * torch.sign(delta_to_wp)) * self.target_generators, axis=0)
        return grad_calc
    
    def constraints(self, slice_lambdas):
        slice_lambdas = slice_lambdas.reshape(-1, 1)
        out = torch.zeros(self.num_constraints)
        for i in range(self.num_constraints):
            cons = self.halfspace_constraints_over_lambda[i]
            cons_val = cons.evaluate(slice_lambdas)
            out[i] = cons_val
        return out

    def jacobian(self, slice_lambdas):
        slice_lambdas = slice_lambdas.reshape(-1, 1)
        out = torch.zeros(self.num_constraints)
        for i in range(self.num_constraints):
            cons = self.halfspace_constraints_over_lambda[i]
            cons_val = cons.evaluate_sub_gradient(slice_lambdas)
            out[i] = cons_val
        return out

def offset_zono(zono, delta, dims):
    z_mat = zono.Z.t()
    z_mat[dims, 0] = z_mat[dims, 0] + delta
    return zonopy.zonotope(z_mat.t())

def zono_get_span_across_dim(z, dim):
    rad = torch.sum(torch.abs(z.generators.t()[dim,:]))
    cen = z.center[dim]
    return [cen - rad, cen + rad]

def generate_constraints(sliced_frs, obs_descriptors_world, param_slice_dimensions, t_dim=-1):
    # FRS sliced on initial condition already
    zono_constraints = []
    for i in range(len(sliced_frs)):
        z_curr = sliced_frs[i]
        obstacles_over_t = obs_descriptors_world.get_xy_zonotopes_over_time_interval(*(zono_get_span_across_dim(z_curr, t_dim)))
        for obs in obstacles_over_t:
            z_curr_gens_indices = find_sliceable_generator_idx(z_curr, param_slice_dimensions)
            z_curr_slice_gens = z_curr.generators.t()[0:2, z_curr_gens_indices]
            z_curr_center_sliced = z_curr.slice(slice_dim=param_slice_dimensions, slice_pt=z_curr.center[param_slice_dimensions])
            z_curr_center_sliced_projected = z_curr_center_sliced.project(slice(0,2))
            delta_cen = (obs.center - z_curr_center_sliced_projected.center).reshape(-1, 1)
            z_curr_center_sliced_buffered = zonopy.zonotope(torch.hstack([delta_cen, z_curr_center_sliced_projected.generators.t(), obs.generators.t()]).t())
            z_curr_center_sliced_buffered_projected = z_curr_center_sliced_buffered.project(slice(0,2))
            PA, Pb = z_curr_center_sliced_buffered_projected.polytope()
            unparameterized_halfspace = Halfspace(PA, Pb)
            slc_G = z_curr_slice_gens
            slc_c = torch.zeros((1, z_curr_slice_gens.shape[1]))
            slc_g = torch.ones((1, z_curr_slice_gens.shape[1]))
            parameterized_halfspace = unparameterized_halfspace.parameterize(slc_G=slc_G, slc_c=slc_c, slc_g=slc_g)
            parameterized_halfspace_simplified = parameterized_halfspace.simplify(torch.hstack([slc_c - torch.abs(slc_g), slc_c + torch.abs(slc_g)]))
            if parameterized_halfspace_simplified.a_mat.numel():
                zono_constraints.append(parameterized_halfspace_simplified)
    return zono_constraints


class SimpleReachSet:
    '''
    Assumes uniform dimensions and indices across all zonotopes
    '''
    def __init__(self, zono_list):
        self.zono_list = zono_list
    
    def num_zonotopes(self):
        return len(self.zono_list)
    
    def zonotope(self, idx):
        return self.zono_list[idx]
    
    def offset_by(self, delta, dims):
        return SimpleReachSet([offset_zono(zono, delta, dims) for zono in self.zono_list])
    
    def rotate_by_radians(self, delta_h, x_dim=0, y_dim=1, h_dim=2):
        return SimpleReachSet([rotate_by_radians(zono, delta_h, x_dim, y_dim, h_dim) for zono in self.zono_list])
    
    def time_interval_n(self, n, time_dim=-1):
        assert False, "FIX FOR TORCH"
        assert (n >= 0 and n < self.num_zonotopes()), "The index is out of bounds."
        sz = self.zonotope(n)
        tzc = sz.center[time_dim]
        tzg = np.sum(np.abs(sz.generators[time_dim, :]))
        return [tzc - tzg, tzc + tzg]
    
    def point_slice(self, dimension, slice_value):
        return SimpleReachSet([zono.slice(slice_dim=dimension, slice_pt=slice_value) for zono in self.zono_list])

    def __iter__(self):
        return iter(self.zono_list)
    
    def __getitem__(self, idx):
        return self.zono_list[idx]
    
    def __len__(self):
        return len(self.zono_list)
