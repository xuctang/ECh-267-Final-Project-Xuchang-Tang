import numpy as np
from scipy.optimize import fsolve
from casadi import *
import matplotlib.pyplot as plt
import os

# directory for figure saving
folder = './Figures'
save_type = '.png'

if not os.path.exists(folder):
    os.makedirs(folder)

def model(x, u, param, xs=None, us=None, noise=True):
    """Compute the right-hand side of the ODEs
    
    Args:
        x (array-like): State vector
        u (array-like): Input vector
        param (dictionary): Parameters
        xs (array-like, optional): steady-state
        us (array-like, optional): steady-state input
    
    Returns:
        array-like: dx/dt    
    """
    if xs is not None:
        # Assume x is in deviation variable form
        x = [x[i] + xs[i] for i in range(2)]
        
    if us is not None:
        # Assume u is in deviation variable form
        u = [u[i] + us[i] for i in range(2)]
        
    # unpacking
    To = param['To']
    p1 = param['p1']
    p2 = param['p2']
    p3 = param['p3']
    p4 = param['p4']
    p5 = param['p5']
    p6 = param['p6']
        
    dxdt = [0.] * 2
    dxdt[0] = p1 * (To - x[0]) - p2 * np.exp(-p3 / x[0]) * x[1]
    dxdt[1] = p4 * (u[0] - x[1]) + p5 * np.exp(-p3 / x[0]) * x[1] + p6 * u[1]
    
    if noise is True:
        for ii in range(2):
            dxdt[ii] += np.random.normal(0, param['sigma'][ii], 1)[0]
    
    return dxdt

def noise_study(model, xs, us, ratio, T, N, dt, param, Q, R, ulb, uub, num_time_steps_sim, num_trials):
    """
    The large wrapper function for convenience in studying the noise effect of MPC
    
    """
    
    def solve_mpc(current_state):
        """Solve MPC provided the current state, i.e., this 
        function is u = h(x), which is the implicit control law of MPC.

        Args:
            current_state (array-like): current state

        Returns:
            tuple: current input and return status pair
        """

        # Set the lower and upper bound of the decision variable
        # such that s0 = current_state
        for i in range(2):
            zlb[i] = current_state[i]
            zub[i] = current_state[i]
        sol_out = solver(lbx=zlb, ubx=zub, lbg=g_bnd, ubg=g_bnd)
        return (np.array(sol_out['x'][2:4]), solver.stats()['return_status'])

    """
    Main block
    """
    # MPC steady state array
    xs_MPC = np.zeros((num_trials, 2))
    
    # Figure plotting
    plt.figure(figsize=[14, 14])
    fig, axs = plt.subplots(4, 1, figsize=[10, 10])
    
    for trial in range(num_trials):
        # CasADi with symbolics
        t = SX.sym("t", 1, 1)
        x = SX.sym("x", 2, 1)
        u = SX.sym("u", 2, 1)
        ode = vertcat(*model(x, u, param, xs=xs, us=us, noise=True))

        # models
        f = {'x': x, 't':t, 'p':u , 'ode':ode}
        Phi = integrator("Phi", "cvodes", f, {'tf': dt})
        ode = vertcat(*model(x, u, param, xs=xs, us=us))
        Phi = integrator("Phi_clear", "cvodes", f, {'tf': dt})
        system = Phi

        # Define the decision variable and constraints
        q = vertcat(*[MX.sym(f'u{i}', 2, 1) for i in range(N)])
        s = vertcat(*[MX.sym(f'x{i}', 2, 1) for i in range(N+1)])
        z = []
        zlb = []
        zub = []
        constraints = []

        # Create a function
        cost = 0.

        for i in range(N):
            s_i = s[2*i:2*(i+1)]
            s_ip1 = s[2*(i+1):2*(i+2)]
            q_i = q[2*i:2*(i+1)]

            # Decision variable
            zlb += [0.0, 0.0]
            zub += [500.0, 50.0]
            zlb += ulb
            zub += uub

            z.append(s_i)
            z.append(q_i)

            xt_ip1 = Phi(x0=s_i, p=q_i)['xf']
            cost += s_i.T @ Q @ s_i + q_i.T @ R @ q_i
            constraints.append(xt_ip1 - s_ip1)

        # s_N
        z.append(s_ip1)
        zlb += [-np.inf] * 2
        zub += [np.inf] * 2

        constraints = vertcat(*constraints)
        variables = vertcat(*z)

        # Create the optmization problem
        g_bnd = np.zeros(N*2)
        nlp = {'f': cost, 'g': constraints, 'x': variables}
        opt = {'print_time': 0, 'ipopt.print_level': 0, 'ipopt.acceptable_tol': 1e-3}
        solver = nlpsol('solver', 'ipopt', nlp, opt)

        # Store the system states and control actions applied to the system
        # in array
        state_history = np.zeros((num_time_steps_sim+1, 2))
        input_history = np.zeros((num_time_steps_sim+1, 2))

        # Set current state - using deviation variables
        state_history[0, :] = np.array([300.0, 50.0]) - xs
        current_state = state_history[0, :]

        # Time array for plotting
        time = [i*dt for i in range(num_time_steps_sim+1)]

        # Closed-loop simulation
        for k in range(num_time_steps_sim):

            print(f'Current time: {k*dt}')
            current_control, status = solve_mpc(current_state)
            print(f'Solver status: {status}')

            # Advance the simulation one time step
            # Set current_state to be the state at the next time steps
            current_state = np.array(system(x0=current_state, p=current_control)['xf'])

            current_state = current_state.reshape((2,))
            current_control = current_control.reshape((2,))

            # Save data for plotting
            input_history[k, :] = current_control
            state_history[k+1:k+2, :] = current_state

        # Save the last control one more time for plotting
        input_history[-1, :] = current_control
        
        # deliverables
        for ii in range(2):
            xs_MPC[trial, ii] = np.mean(state_history[-5:-1, ii])

        # Plotting
        t_max = min(5, num_time_steps_sim*dt)
        for j in range(2):
            axs[j].plot(time, state_history[:, j]+xs[j], 'r-', alpha=0.5)
            axs[j].plot([time[0], time[-1]], [xs[j], xs[j]], 'r--')
            axs[j].plot([time[0], time[-1]], [xs[j] * (1 + ratio), xs[j] * (1 + ratio)], 'b--', alpha=0.7)
            axs[j].plot([time[0], time[-1]], [xs[j] * (1 - ratio), xs[j] * (1 - ratio)], 'b--', alpha=0.7)
            axs[j].set_ylabel(f'$x_{j+1}$')
            axs[j].set_xlim([0, t_max])

        for j in range(2):
            axs[j+2].step(time, input_history[:, j]+us[j], 'r-', alpha=0.5, where='post')
            axs[j+2].step([time[0], time[-1]], [us[j], us[j]], 'r--', alpha=0.5, where='post')
            axs[j+2].set_ylabel(f'$u_{j+1}$')
            axs[j+2].set_xlim([0, t_max])
            
    axs[-1].set_xlabel('Time')
    ttl = f"MPC Solution for the Chemical Reaction Problem with Uncertainty Ratio Q = {round(ratio, 2)}"
    axs[0].set_title(ttl)

    # save
    save_path = folder + '/' + ttl + save_type
    plt.savefig(save_path)
    
    # deliverables
    # remaining state deviation
    xs_Q_1 = np.mean(xs_MPC[:, 0])
    xs_Q_2 = np.mean(xs_MPC[:, 1])
    # summed errors
    se_1 = np.sum(np.abs(xs_MPC[:, 0]) / xs[0])
    se_2 = np.sum(np.abs(xs_MPC[:, 1]) / xs[1])
    
    return xs_Q_1, xs_Q_2, se_1, se_2


if __name__ == '__main__':
    # Parameters (used in optimal control problem later as well)
    T = 2.0
    N = 15
    dt = T/N

    num_time_steps_sim = 15  # number of time steps in simulation

    param = {'To': 300, 'p1': 2, 'p2': 2, 'p3': 5, 'p4': 5, 'p5': 2, 'p6': 100}
    # Get the steady-state
    us = np.array([2.5, 0.5])
    f = lambda x: model(x, us, param, noise=False)
    xs, _, flag, _ = fsolve(f, [200, 200], full_output=True)

    print(f'xs = {xs}')
    print(f'Exit flag: {flag}')

    # constrains
    Q = np.array([[5.0, 2.5],
                [2.5, 5.0]])
    R = np.array([[1.25, 0.5], 
                [0.5, 1.25]])
    ulb = list(np.array([0.0, -10.0]))
    uub = list(np.array([25.0, 10.0]))

    ratio_arr = [0.0, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.5]
    #ratio_arr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    num_trials = 10  # number of trials for each ratio
    xs_Q_1 = [0.] * len(ratio_arr)
    xs_Q_2 = [0.] * len(ratio_arr)
    err_Q_1 = [0.] * len(ratio_arr)
    err_Q_2 = [0.] * len(ratio_arr)
    se_1 = [0.] * len(ratio_arr)
    se_2 = [0.] * len(ratio_arr)
    se = [0.] * len(ratio_arr)

    for rr, ratio in enumerate(ratio_arr):
        print('*****************************')
        print(f"Starting to solve for Q = {round(ratio, 2)}")
        print('*****************************')
        param['sigma'] = [ratio * xs[ii] for ii in range(2)]
        err_Q_1[rr], err_Q_2[rr], se_1[rr], se_2[rr] = noise_study(model, xs, us, ratio, T, N, dt, param, Q, R, ulb, uub, num_time_steps_sim, num_trials)

    print('Loop finished!')

    for rr in range(len(ratio_arr)):
        xs_Q_1[rr] = err_Q_1[rr] + xs[0]
        xs_Q_2[rr] = err_Q_2[rr] + xs[1]

        se[rr] = se_1[rr] + se_2[rr]

    # plotting for summed error
    plt.figure(figsize=(12, 6))
    plt.plot(ratio_arr, se, 'k-', label="Summed Percentage Error")
    ttl = "Summed Percentage Error against Q Ratios"
    plt.title(ttl)
    plt.xlabel("Q Ratio")
    plt.ylabel("Summed Percentage Error over Trials")
    save_path = folder + "/" + ttl + save_type
    plt.savefig(save_path)

    # plotting for overall xs_Q versus Q
    plt.figure(figsize=[14, 14])
    fig, axs = plt.subplots(4, 1, figsize=[10, 10])
            
    # state 1
    axs[0].plot(ratio_arr, xs_Q_1, 'k-', label='Value with noise') 
    axs[0].plot([ratio_arr[0], ratio_arr[-1]], [xs[0], xs[0]], 'b--', label='Target value')
    axs[0].set_ylabel(f"Steady state $xs_1$")
    axs[0].legend()
                            
    # state 2
    axs[1].plot(ratio_arr, xs_Q_2, 'k-', label='Value with noise') 
    axs[1].plot([ratio_arr[0], ratio_arr[-1]], [xs[1], xs[1]], 'b--', label='Target value')
    axs[1].set_ylabel(f"Steady state $xs_2$")
    axs[1].legend()

    # error 1
    axs[2].plot(ratio_arr, np.abs(err_Q_1), 'k-', label='Error due to noise') 
    axs[2].plot([ratio_arr[0], ratio_arr[-1]], [0.0, 0.0], 'b--')
    axs[2].set_ylabel(f"Error $e_1$")
    axs[2].legend()
                                
    # error 2
    axs[3].plot(ratio_arr, np.abs(err_Q_2), 'k-', label='Error due to noise') 
    axs[3].plot([ratio_arr[0], ratio_arr[-1]], [0.0, 0.0], 'b--')
    axs[3].set_ylabel(f"error $e_2$")
    axs[3].legend()
                            
    axs[-1].set_xlabel("Q ratio")
    ttl = "Noise effect on MPC performance"
    axs[0].set_title(ttl)

    # save
    save_path = folder + '/' + ttl + save_type
    plt.savefig(save_path)

