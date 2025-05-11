# Snippet 1: Imports & Problem Parameters
import numpy as np
import random
import matplotlib.pyplot as plt
import math  # Import math for log2 and ceil

# Jobs, operations, factories
J, O = 20, 9
W, Wp = 5, 3  # total factories, number of PF factories
machines_per_factory = {
    w: ([1] * O if w < Wp else [random.randint(2, 5) for _ in range(O)])
    for w in range(W)
}

# Base energy parameters
energy_per_operation = np.random.randint(1, 11, size=O)  # Ulmh base per operation
energy_per_machine = {
    w: ([1] * O if w < Wp else [random.randint(1, 5) for _ in range(O)])
    for w in range(W)
}  # Unmh per operation

robot_energy_per_unit_time = 0.1  # for robot travel

# Derived energy rates per machine
U_l = {  # loaded energy rate per (w,op)
    w: [energy_per_operation[op] * energy_per_machine[w][op] for op in range(O)]
    for w in range(W)
}
U_nm = energy_per_machine  # no-load energy rate per (w,op)

# On/off switching cost and shut‐down threshold SV
E_onoff = 0.67  # energy cost to switch a machine off/on
SV = {
    (w, op, k): E_onoff / U_nm[w][op]
    for w in range(W)
    for op in range(O)
    for k in range(machines_per_factory[w][op])
}

# Processing times, transport times, release times
pt = np.random.randint(10, 101, size=(J, O))
t_trans = np.random.randint(1, 21, size=O - 1)
RT = np.random.randint(1, 6, size=J)

# Snippet 2: SFL-DEA Hyperparameters
NP, maxI = 50, 100  # population size and maximum iterations
uf, uCR = 0.5, 0.3  # initial scaling factor (F) and crossover probability (CR)
beta, mu = 3, 20  # number of memeplexes and maximum local search iterations

# Snippet 3: Individual Encoding & Initialization
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def init_population_good_points():
    # Find h: smallest prime such that (h-3)/2 >= J
    h = 5
    while not (is_prime(h) and (h - 3) // 2 >= J):
        h += 2

    # Initialize population list
    population = []
    # Base scheduling vector
    y0 = list(range(J))

    # Compute good-point set X
    X = []
    for p in range(1, NP + 1):
        xp = [p * 2 * math.cos(2 * math.pi * j / h) for j in range(J)]
        X.append(xp)

    # Map X to target space and generate individuals
    for p, xp in enumerate(X, start=1):
        Ubp = max(xp)
        Lbp = min(xp)
        xp_prime = [(Lbp + (x - math.floor(x)) * (Ubp - Lbp)) for x in xp]

        # Sort base y0 by xp_prime to get scheduling vector yp
        yp = [job for _, job in sorted(zip(xp_prime, y0))]

        # Factory assignment sub-group via RP
        # Ensure each factory has at least one job
        base = list(range(W))
        extra = [random.randrange(W) for _ in range(J - W)]
        rho = base + extra
        random.shuffle(rho)

        population.append({'y': yp, 'rho': rho})

    return population


# Snippet 4: Q-Learning Agent and Related Functions
class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Explore: Choose a random action
            return random.randrange(self.q_table.shape[1])
        else:
            # Exploit: Choose the best action based on Q-values
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        # Update Q-value using the Q-learning formula
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def get_state(self, sched, idle, failed_machines):
        # ... (Your get_state implementation) ...
        # 1. Machine Availability:
        machine_availability = []
        for w in range(W):  # For each factory
            for op in range(O):  # For each operation
                for k in range(machines_per_factory[w][op]):  # For each machine
                    machine_availability.append(1 if (w, op, k) not in failed_machines else 0)
        # 2. Job Priorities (Example - You might need to adapt this based on your problem):
        job_priorities = [0] * J  # Initialize with default priority here all jobs are considered to be of equal priority
        # ... (Logic to assign priorities based on due dates, processing times, etc.) ...
        # 3. Remaining Processing Times:
        '''Discretization of the remaining times'''
        remaining_times = []
        max_remaining_time = (pt.max())  # Get the maximum possible remaining time
        num_bins = 2**bits_for_remaining_time # Calculate the number of bins #You need to define bits_for_remaining_time
        bin_width = max_remaining_time / num_bins  # Calculate the width of each bin

        for j in range(J):
            remaining_time_for_job = 0
            for op in range(O):
                if (j, op) not in sched:  # Operation not yet scheduled
                    remaining_time_for_job += pt[j, op]

            # Discretize remaining_time_for_job into bins
            discretized_remaining_time = int(remaining_time_for_job / bin_width)
            discretized_remaining_time = min(discretized_remaining_time, num_bins - 1) #Preventing the overflow in an array
            remaining_times.append(discretized_remaining_time)
        # Combine features into a state representation
        state_features = machine_availability + job_priorities + remaining_times

        # Convert to a state index (integer) using binary encoding:
        state_index = 0
        for i, feature in enumerate(state_features):
            state_index |= (feature << i)  # Bitwise OR to combine features

        # Ensure state_index is within the Q-table size:
        state_index = state_index % self.q_table.shape[0]
        return state_index

    def get_reward(self, sched, rho):
        # ... (Your get_reward implementation) ...
        # 1) Makespan
        all_jobs_scheduled = all((j, O - 1) in sched for j in range(J))
    
        if all_jobs_scheduled:
          makespan = max(sched[(j, O - 1)][1] for j in range(J))
        else:
        # Assign a large penalty if not all jobs are scheduled
          makespan = float('inf')  # or a very large number

        # 2) Energy
        total_energy = 0.0

        # Track last finish per machine to compute idle gaps
        last_finish = {(w, op, k): 0
                       for w in range(W)
                       for op in range(O)
                       for k in range(machines_per_factory[w][op])}

        for (j, op), (start, end, w, k) in sched.items():
            # a) idle gap energy (no-load vs on/off)
            gap = start - last_finish[(w, op, k)]
            if gap >= SV[(w, op, k)]:
                total_energy += E_onoff
            else:
                total_energy += U_nm[w][op] * gap

            # b) load energy
            total_energy += U_l[w][op] * (end - start)

            # c) robot travel energy
            if op > 0:
                total_energy += robot_energy_per_unit_time * t_trans[op - 1]

            # update last finish
            last_finish[(w, op, k)] = end

        # 3) Combine into reward (negative for minimization)
        return -(makespan + total_energy)


# Snippet 4 (continued): Decode Function
def decode(ind, failure_prob=0.01, td=10, agent=None):
    # ... (Your decode implementation) ...
    # Initialize RL agent if not passed in
    total_machines = sum(sum(machines_per_factory[w][op] for op in range(O)) for w in range(W))
    bits_priority = math.ceil(math.log2(J))
    bits_remaining = math.ceil(math.log2(pt.max()))
    state_size = total_machines + bits_priority * J + bits_remaining * J
    action_size = total_machines
    if agent is None:
        agent = QLearningAgent(0.1, 0.9, 0.9, state_size, action_size)

    y, rho = ind['y'], ind['rho']
    idle = {(w, op, k): 0
            for w in range(W)
            for op in range(O)
            for k in range(machines_per_factory[w][op])}
    failed_machines = set()
    sched = {}
    total_energy = 0.0

    for op in range(O):
        seq = y if op == 0 else sorted(y, key=lambda j: sched[(j, op - 1)][1])
        for j in seq:
            w = rho[j]
            mcount = machines_per_factory[w][op]

            # simulate failure
            if random.random() < failure_prob:
                fm = (w, op, random.randrange(mcount))
                failed_machines.add(fm)
                idle[fm] += td

            # decide machine k
            if agent:
                state = agent.get_state(sched, idle, failed_machines)
                action = agent.choose_action(state)
                k = action % mcount
            else:
                avail = [kk for kk in range(mcount) if (w, op, kk) not in failed_machines]
                k = min(avail, key=lambda kk: idle[(w, op, kk)]) if avail else 0

            # compute candidate start
            prev_end = sched[(j, op - 1)][1] if op > 0 else RT[j]
            travel = t_trans[op - 1] if op > 0 else 0
            ready = prev_end + travel
            last_fin = idle[(w, op, k)]
            start_cand = max(last_fin, ready)

            # on/off vs no-load energy
            gap = start_cand - last_fin
            if gap >= SV[(w, op, k)]:
                total_energy += E_onoff
            else:
                total_energy += U_nm[w][op] * gap

            # load energy
            start = start_cand
            end = start + pt[j][op]
            total_energy += U_l[w][op] * (end - start)

            # robot travel energy
            if op > 0:
                total_energy += robot_energy_per_unit_time * t_trans[op - 1]

            # record schedule
            sched[(j, op)] = (start, end, w, k)  # Modified to include w, k
            idle[(w, op, k)] = end

            # online RL update
            if agent:
                reward = agent.get_reward(sched, rho)
                next_state = agent.get_state(sched, idle, failed_machines)
                agent.update_q_table(state, action, reward, next_state)

    ind['sched'] = sched
    # Calculate Cmax only if all jobs are scheduled
    all_jobs_scheduled = all((j, O - 1) in sched for j in range(J))
    if all_jobs_scheduled:
        ind['Cmax'] = max(sched[(j, O - 1)][1] for j in range(J))
    else:
        ind['Cmax'] = float('inf')  # or a very large penalty value
    ind['TEC'] = total_energy
    ind['failed_machines'] = list(failed_machines)
    return ind


# Snippet 4 (continued): Train RL Agent Function
def train_rl_agent(agent, num_episodes, failure_prob=0.1, td=10):
    # ... (Your train_rl_agent implementation) ...
    for episode in range(num_episodes):
        # Generate a random initial individual (schedule)
        # ind = init_population_good_points() # Assuming you have an init_individual() function
        # Changed to get only one individual, not a population
        ind = init_population_good_points()[0]
        # Simulate the scheduling process with potential failures and RL agent
        ind = decode(ind, failure_prob=failure_prob, td=td, agent=agent)

        # You can print or log the progress periodically
        if episode % 100 == 0:
            print(f"Episode: {episode}, Makespan: {ind['Cmax']}, Energy: {ind['TEC']}")

    print("RL agent training completed.")


# Snippet 5: Fast Non‐Dominated Sorting & Crowding Distance
def fast_nondom_sort(pop):
    # ... (Your fast_nondom_sort implementation) ...
    S = {i: [] for i in range(len(pop))}
    n = {i: 0 for i in range(len(pop))}
    fronts = []
    # Compare each pair of solutions
    for p in range(len(pop)):
        for q in range(len(pop)):
            cp, cq = pop[p]['Cmax'], pop[q]['Cmax']
            ep, eq = pop[p]['TEC'], pop[q]['TEC']
            if (cp <= cq and ep <= eq) and (cp < cq or ep < eq):
                S[p].append(q)
            elif (cq <= cp and eq <= ep) and (cq < cp or eq < ep):
                n[p] += 1
        # Initialize 'rank' for all individuals to a default value (e.g., infinity)
        pop[p]['rank'] = float('inf')  # Initialize rank to infinity
        if n[p] == 0:
            pop[p]['rank'] = 1  # Assign rank 1 if not dominated
    # Build Pareto fronts
    current = [i for i in range(len(pop)) if pop[i]['rank'] == 1]
    while current:
        fronts.append(current)
        nxt = []
        for p in current:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q]['rank'] = pop[p]['rank'] + 1
                    nxt.append(q)
        current = nxt
    return fronts


def crowding_distance(front, pop):
    # ... (Your crowding_distance implementation) ...
    l = len(front)
    for idx in front:
        pop[idx]['cd'] = 0
    for key in ['Cmax', 'TEC']:
        front.sort(key=lambda i: pop[i][key])
        pop[front[0]]['cd'] = pop[front[-1]]['cd'] = 1e9
        vals = [pop[i][key] for i in front]
        if max(vals) - min(vals) < 1e-6:  # Tolerance for identical values
            # Skip division or assign a default crowding distance
            continue
        else:
            for k in range(1, l - 1):
                pop[front[k]]['cd'] += (vals[k + 1] - vals[k - 1]) / (max(vals) - min(vals) + 1e-9)
   


# Snippet 6: Binary Tournament Selection
def tournament_selection(pop):
    # ... (Your tournament_selection implementation) ...
    selected = []
    while len(selected) < NP:
        a, b = random.sample(range(len(pop)), 2)
        pa, pb = pop[a], pop[b]
        # Compare by rank, then crowding distance
        if pa['rank'] < pb['rank'] or (pa['rank'] == pb['rank'] and pa['cd'] > pb['cd']):
            selected.append(dict(pa))
        else:
            selected.append(dict(pb))
    return selected

#self adaptive DE
def mutate(pop):
    # ... (Your mutate implementation) ...
    best = min(pop, key=lambda x: x['rank'])
    mutants = []
    for ind in pop:
        # 1. Sample F and CR
        F = max(0, min(1, random.gauss(uf, 0.1)))
        CR = max(0, min(1, random.gauss(uCR, 0.1)))

        # 2. Choose a DEM combination
        if random.random() < 0.5:
            # Strategy 1: DE/best/1 then DE/rand/1
            a, b = random.sample(pop, 2)
            # Apply DE/best/1
            # Check for empty lists before accessing elements
            best_y_len = len(best['y']) if best['y'] else 0
            ind_y_len = len(ind['y']) if ind['y'] else 0
            
            v1 = [int((best['y'][i] + F * (best['y'][i] - ind['y'][i])) % J) 
                    for i in range(min(best_y_len, ind_y_len))] 
            # Apply DE/rand/1 to v1
            v1_len = len(v1) if v1 else 0
            a_y_len = len(a['y']) if a['y'] else 0
            b_y_len = len(b['y']) if b['y'] else 0
            
            v2 = [int((v1[i] + F * (a['y'][i] - b['y'][i])) % J) 
                    for i in range(min(v1_len, a_y_len, b_y_len))] 
            y_mut = v2
        else:
            # Strategy 2: DE/best/2 then DE/rand/2
            a, b, c, d, e, f_ = random.sample(pop, 6)
            
            # Check for empty lists before accessing elements
            best_y_len = len(best['y']) if best['y'] else 0
            a_y_len = len(a['y']) if a['y'] else 0
            b_y_len = len(b['y']) if b['y'] else 0
            c_y_len = len(c['y']) if c['y'] else 0
            d_y_len = len(d['y']) if d['y'] else 0
            
            # DE/best/2: best + F*(a-b) + F*(c-d)
            v1 = [int((best['y'][i] + F * (a['y'][i] - b['y'][i]) + F * (c['y'][i] - d['y'][i])) % J) 
                  for i in range(min(best_y_len, a_y_len, b_y_len, c_y_len, d_y_len))]
            
            ind_y_len = len(ind['y']) if ind['y'] else 0
            e_y_len = len(e['y']) if e['y'] else 0
            f__y_len = len(f_['y']) if f_['y'] else 0            
            
            # DE/rand/2: ind + F*(e-f) + F*(g-h)
            v2 = [int((ind['y'][i] + F * (e['y'][i] - f_['y'][i]) + F * (c['y'][i] - d['y'][i])) % J) 
                  for i in range(min(ind_y_len, e_y_len, f__y_len, c_y_len, d_y_len))]
            
            v1_len = len(v1) if v1 else 0
            v2_len = len(v2) if v2 else 0
            
            # Combine results
            y_mut = [int((v1[i] + v2[i]) % J) for i in range(min(v1_len, v2_len))]

        # 3. Binomial crossover with CR
        y_mut_len = len(y_mut) if y_mut else 0
        ind_y_len = len(ind['y']) if ind['y'] else 0
        
        child_y = [y_mut[i] if random.random() < CR else ind['y'][i] 
                for i in range(min(y_mut_len, ind_y_len))]

        mutants.append({'y': child_y, 'rho': ind['rho'][:]})

    return mutants

# Snippet 8: Bidirectional Crossover Strategy
def crossover(pop):
    # ... (Your crossover implementation) ...
    offspring = []
    for ind in pop:
        mate = random.choice(pop)
        # Single cut-point
        r = random.randint(1, J - 1)
        p1, p2 = ind['y'], mate['y']
        # Create child by prefix from p1 and fill from p2, ensuring uniqueness
        child_y = p1[:r]
        child_y.extend([x for x in p2 if x not in child_y])
        offspring.append({'y': child_y, 'rho': ind['rho'][:]})
    return offspring


# Snippet 9: Main Optimization Loop with Detailed Energy Model
# ... (Your main optimization loop) ...
# Define bits_for_remaining_time
bits_for_remaining_time = 7  # Assuming 7 bits for the remaining time representation

# 1. Train or load RL agent (offline)
total_machines = sum(
    sum(machines_per_factory[w][op] for op in range(O))
    for w in range(W)
)
bits_priority = math.ceil(math.log2(J))
bits_remaining = math.ceil(math.log2(pt.max()))
state_size = total_machines + bits_priority * J + bits_remaining * J
action_size = total_machines

agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1,
                       state_size=state_size, action_size=action_size)
train_rl_agent(agent, num_episodes=1000)
np.save('q_table.npy', agent.q_table)

# 2. Initialize population using good-point set
population = init_population_good_points()  # Changed to init_population_good_points

# 3. Evolutionary loop
for gen in range(maxI):
    # a) Decode & evaluate each individual with detailed energy + RL
    for ind in population:
        ind = decode(ind, failure_prob=0.01, td=10, agent=agent)

    # b) Non-dominated sorting & crowding
    fronts = fast_nondom_sort(population)
    for f in fronts:
        crowding_distance(f, population)

    # c) Selection → Mutation → Crossover
    selected = tournament_selection(population)
    mutants = mutate(selected)  # self-adaptive DE with CR sampling
    offspring = crossover(mutants)  # bidirectional
    # d) Decode offspring
    for ind in offspring:
        ind = decode(ind, failure_prob=0.01, td=10, agent=agent)

    # e) Merge & re-select
    merged = population + offspring
    fronts = fast_nondom_sort(merged)
    new_pop = []
    for f in fronts:
        crowding_distance(f, merged)
        for idx in f:
            if len(new_pop) < NP:
                new_pop.append(merged[idx])
    population = new_pop

    # f) (Optional) adapt uf, uCR per your strategy

# 4. Extract and display Pareto front
pareto_front = [population[i] for i in fast_nondom_sort(population)[0]]



# Snippet 10: Display Results
# ... (Your display results implementation, using pareto_front from Snippet 9) ...
# Get the best solution (from\
# Define weights for Makespan and Energy Consumption (e.g., 0.5 for each)
makespan_weight = 0.5
energy_weight = 0.5

# Calculate a weighted score for each solution
for solution in pareto_front:
    solution['weighted_score'] = (makespan_weight * solution['Cmax'] + 
                                 energy_weight * solution['TEC'])

# Find the solution with the lowest weighted score
best_balanced_solution = min(pareto_front, key=lambda x: x['weighted_score'])

print("Best Balanced Solution:")
print("Makespan:", best_balanced_solution['Cmax'])
print("Energy:", best_balanced_solution['TEC'])
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_chart(schedule, J, O, W):
    """Plots a Gantt chart for the given schedule.

    Args:
        schedule (dict): The schedule dictionary.
        J (int): Number of jobs.
        O (int): Number of operations.
        W (int): Number of factories.
    """

    # Define colors for each operation (you can customize these)
    operation_colors = plt.cm.get_cmap('tab10', O)  # Using 'tab10' colormap

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))  

    # Plot tasks
    for (job, op), (start, end, factory, machine) in schedule.items():
        ax.barh(job, end - start, left=start, height=0.8, 
                color=operation_colors(op), edgecolor='black')
        # Add text for factory at the end of the task
        ax.text(end, job, f"F{factory+1}", va='center', ha='left', fontsize=8) 

    # Customize chart
    ax.set_xlabel("Time")
    ax.set_ylabel("Job")
    ax.set_title("Job Shop Schedule Gantt Chart")
    ax.set_yticks(range(J))  # Set y-ticks to job numbers
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Create legend for operations
    legend_patches = [mpatches.Patch(color=operation_colors(op), label=f"Operation {op+1}")
                      for op in range(O)]
    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()
# Assuming 'best_solution' is the selected solution from the Pareto front
plot_gantt_chart(best_balanced_solution['sched'], J, O, W)
# Assuming 'best_balanced_solution' is your selected solution
failed_machines = best_balanced_solution['failed_machines']

if failed_machines:
    print("Failed Machines:")
    for machine in failed_machines:
        factory, operation, machine_id = machine  # Unpack the tuple
        print(f"  - Factory {factory + 1}, Operation {operation + 1}, Machine {machine_id + 1}")
else:
    print("No machines failed during this schedule.")





