from refine_backend import *
import matplotlib.pyplot as plt

def construct_test_zono(
    center_x, center_y, center_h,
    xyh_gens_nonslice,
    sliceable_val_infos,
    num_rows = 20,
):
    max_dim_from_entries = torch.max(torch.tensor([entry['dim'] for entry in sliceable_val_infos] + [0]))
    assert max_dim_from_entries < num_rows, "The maximum dimension from the entries is greater than the maximum dimension."
    # {'dim', 'gen_slc_dim', 'center_slc_dim', 'xyh_gen'}
    c = torch.zeros((num_rows, 1))
    xyh_dims = slice(0, 3)
    c[xyh_dims] = torch.tensor([[center_x], [center_y], [center_h]])
    gens_slc = torch.zeros((num_rows, len(sliceable_val_infos)))
    for i in range(len(sliceable_val_infos)):
        entry = sliceable_val_infos[i]
        entry_dim = entry['dim']
        c[entry_dim] = entry['center_slc_dim']
        gens_slc[xyh_dims, [i]] = entry['xyh_gen']
        gens_slc[entry_dim, [i]] = entry['gen_slc_dim']
    gens_non_slc = torch.zeros((num_rows, xyh_gens_nonslice.shape[1]))
    gens_non_slc[xyh_dims, :] = xyh_gens_nonslice
    return zonopy.zonotope(torch.hstack([c, gens_non_slc, gens_slc]).t())


def construct_test_zono_set_torch(
        u0_dim = 13, u0_cen = 3, u0_gen = 1, 
        v0_dim = 14, v0_cen = 3, v0_gen = 1, 
        r0_dim = 15, r0_cen = 3, r0_gen = 1, 
        p0_dim = 11, p0_cen = 0, p0_gen = 2, 
        t_dim = -1,
    ):

    z_test1 = construct_test_zono(
        center_x = 0, center_y = 3.5, center_h = 2.5,
        xyh_gens_nonslice=torch.tensor([
            [10.0, 0.0, 0.5],
            [0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0],
        ]), 
        sliceable_val_infos=[
            {'dim': u0_dim, 'gen_slc_dim': u0_gen, 'center_slc_dim': u0_cen, 'xyh_gen': torch.tensor([[1.0], [2], [3]])},
            {'dim': v0_dim, 'gen_slc_dim': v0_gen, 'center_slc_dim': v0_cen, 'xyh_gen': torch.tensor([[4.0], [5], [6]])},
            {'dim': r0_dim, 'gen_slc_dim': r0_gen, 'center_slc_dim': r0_cen, 'xyh_gen': torch.tensor([[7.0], [8], [9]])},
            {'dim': p0_dim, 'gen_slc_dim': p0_gen, 'center_slc_dim': p0_cen, 'xyh_gen': torch.tensor([[10.0], [0], [0]])},
            {'dim': t_dim, 'gen_slc_dim': 1, 'center_slc_dim': 1, 'xyh_gen': torch.tensor([[3.0], [0], [0]])},
        ], 
        num_rows=20,
    )
    z_test2 = construct_test_zono(
        center_x = 0, center_y = 60.5, center_h = 2.5,
        xyh_gens_nonslice=torch.tensor([
            [3.4, 0.0, 0.3],
            [0.0, 1.2, -1.0],
            [0.0, 0.0, 0.0],
        ]),
        sliceable_val_infos=[
            {'dim': u0_dim, 'gen_slc_dim': u0_gen, 'center_slc_dim': u0_cen, 'xyh_gen': torch.tensor([[1.0], [2], [3]])},
            {'dim': v0_dim, 'gen_slc_dim': v0_gen, 'center_slc_dim': v0_cen, 'xyh_gen': torch.tensor([[4.0], [5], [6]])},
            {'dim': r0_dim, 'gen_slc_dim': r0_gen, 'center_slc_dim': r0_cen, 'xyh_gen': torch.tensor([[7.0], [8], [9]])},
            {'dim': p0_dim, 'gen_slc_dim': p0_gen, 'center_slc_dim': p0_cen, 'xyh_gen': torch.tensor([[10.0], [0], [0]])},
            {'dim': t_dim, 'gen_slc_dim': 2, 'center_slc_dim': 4, 'xyh_gen': torch.tensor([[5.0], [0], [0]])},
        ], 
        num_rows=20,
    )
    return SimpleReachSet([z_test1, z_test2])

def reach_set_plot(reach_set, project_dimensions=slice(0,2), color='b', linestyle='-', alpha=1.0):
    for z in reach_set:
        z.project(project_dimensions).plot(ax=plt.gca(), edgecolor=color)

def main():
    obs_descriptors_world = Simple2dObstacleSet([
        # Simple2dObstacle(x0 = 0, y0 = 0, h0 = 0, vel = 1, length = 1, width = 1),
        Simple2dObstacle(x0 = 65, y0 = 90, h0 = math.pi + math.pi / 5, vel = 10, length = 30, width = 2),
        Simple2dObstacle(x0 = 30, y0 = -25, h0 = math.pi / 2, vel = 10, length = 30, width = 2),
    ])

    initial_pose = torch.tensor([[0], [0], [0]])
    test_frs_set_body = construct_test_zono_set_torch()
    test_frs_set_world = test_frs_set_body.rotate_by_radians(initial_pose[2], x_dim=0, y_dim=1, h_dim=2).offset_by(initial_pose[0:1], slice(0,2))
    plt.figure(1)
    plt.axis('equal')
    test_frs_set_world_sliced_init_cond = test_frs_set_world.point_slice(dimension=13, slice_value=3.9)
    zono_constraints = []
    param_slice_dimensions = [11]
    zono_constraints = generate_constraints(test_frs_set_world_sliced_init_cond, obs_descriptors_world, param_slice_dimensions)
    target_zono_idx = 1
    target_zono = test_frs_set_world_sliced_init_cond[target_zono_idx]
    target_center = target_zono.center[0:3].reshape(-1, 1)
    sliceable_generator_indices = [find_sliceable_generator_idx(target_zono, i).item() for i in param_slice_dimensions]

    target_generators = target_zono.generators.t()[0:3, sliceable_generator_indices]

    lambda_optimal = torch.tensor([[1.0]])
    wp = target_center + target_generators @ lambda_optimal

    rfp = RefineOptProblem(
        target_center=target_center,
        target_generators=target_generators,
        waypoint=wp,
        halfspace_constraints_over_lambda=zono_constraints,
    )
    rfp.add_option('linear_solver', 'ma57')
    rfp.add_option('max_iter', 1000)
    rfp.add_option('print_level', 4)
    # rfp.add_option('gradient_approximation', 'finite-difference-values')
    # rfp.add_option('jacobian_approximation', 'finite-difference-values')
    rfp.add_option('hessian_approximation', 'limited-memory')
    rfp.add_option('derivative_test', 'first-order')
    x0 = torch.tensor([[-1.0]])
    sln, info = rfp.solve(x0)
    print(f'{sln = }')
    print(f'{info = }')
    
    # Toggle for plotting
    # test_frs_set_world_sliced_final = test_frs_set_world_sliced_init_cond.point_slice(dimension=11, slice_value=sln.item() * param_slice_gen_values.item() + param_slice_cen_values.item())
    # test_frs_set_world_sliced_final_opt = test_frs_set_world_sliced_init_cond.point_slice(dimension=11, slice_value=lambda_optimal.item() * param_slice_gen_values.item() + param_slice_cen_values.item())
    # reach_set_plot(test_frs_set_world_sliced_init_cond, project_dimensions=slice(0,2), color='r', linestyle='--', alpha=1.0)
    # reach_set_plot(test_frs_set_world_sliced_final, project_dimensions=slice(0,2), color='g', linestyle='-', alpha=1.0)
    # plt.plot()
    # plt.show()
main()
