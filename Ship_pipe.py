import numpy as np
import random
import simpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------
# Parameters & Data Initialization
# -----------------------------
W = 2              # Number of factories
J = 20             # Number of ship pipes (jobs)
O = 9              # Number of operations/stages
population_size = 100  # Population size for optimization
max_iterations = 400   # Maximum generations

# Processing times: pt[j, h] ~ U[10, 100]
pt = np.random.randint(10, 101, size=(J, O))
num_machines = 2   # Assume each operation has 2 machines
# Energy consumption for load processing and no-load processing
Ulm = np.random.randint(15, 41, size=(O, num_machines))  # kW for load operation
Unm = np.random.randint(5, 11, size=(O, num_machines))   # kW for no-load operation

# Transportation times and return times (U[1,20])
th = np.random.randint(1, 21, size=(O - 1))
uh = np.random.randint(1, 21, size=(O - 1))

# Additional parameters from the research paper
Eth_on_off = 0.67   # Energy consumption for turning a machine on/off (kW)
Ulrt = 2            # Unit energy consumption for load transportation (kW)
Unrn = 1            # Unit energy consumption for no-load return (kW)
H_big = 1e6         # Big-M constant
SV = 50             # Idle time threshold

# -----------------------------
# Candidate Representation and Initialization
# -----------------------------
def initialize_individual():
    schedule = np.random.permutation(J)
    factory = np.random.randint(0, W, size=J)
    return {'schedule': schedule, 'factory': factory}

def initialize_population(NP):
    return [initialize_individual() for _ in range(NP)]

# -----------------------------
# Decoding and Energy Computation
# -----------------------------
def decode_individual(individual):
    schedule = individual['schedule']
    factory = individual['factory']
    comp_times = np.zeros(J)
    TElm = 0  # Energy from load operations
    TEnlm = 0 # Energy from no-load (idle) operations
    TEom = 0  # Energy for turning machines on/off
    TEtr = 0  # Energy for load transportation
    TEntr = 0 # Energy for no-load return
    # Process jobs factory-wise sequentially
    for f in range(W):
        jobs = [j for j in schedule if factory[j] == f]
        time = 0
        for j in jobs:
            start_time = time
            # Process each operation
            for op in range(O):
                proc_time = pt[j, op]
                time += proc_time
                TElm += Ulm[op, 0] * proc_time  # Load energy consumption
                # If idle time exists between jobs, add no-load energy and on/off cost
                if op == 0 and start_time - 0 > SV:
                    idle = start_time - 0
                    TEnlm += Unm[op, 0] * idle
                    TEom += Eth_on_off
                if op < O - 1:
                    time += th[op]
                    TEtr += Ulrt * th[op]
                    TEntr += Unrn * uh[op]
            comp_times[j] = time
    makespan = np.max(comp_times)
    total_energy = TElm + TEnlm + TEom + TEtr + TEntr
    return makespan, total_energy

def evaluate_individual(individual):
    return decode_individual(individual)

# -----------------------------
# Evolutionary Operators
# -----------------------------
def non_dominated_sort(population):
    scores = [evaluate_individual(ind) for ind in population]
    n = len(population)
    domination_count = [0] * n
    dominated = [set() for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            fi, fj = scores[i], scores[j]
            if (fi[0] <= fj[0] and fi[1] <= fj[1]) and (fi[0] < fj[0] or fi[1] < fj[1]):
                dominated[i].add(j)
            elif (fj[0] <= fi[0] and fj[1] <= fi[1]) and (fj[0] < fi[0] or fj[1] < fi[1]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)
    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts, scores

def crowding_distance(front, scores):
    distance = {i: 0 for i in front}
    for m in range(2):
        sorted_front = sorted(front, key=lambda i: scores[i][m])
        distance[sorted_front[0]] = float('inf')
        distance[sorted_front[-1]] = float('inf')
        for k in range(1, len(sorted_front)-1):
            if scores[sorted_front[-1]][m] - scores[sorted_front[0]][m] == 0:
                dist = 0
            else:
                dist = (scores[sorted_front[k+1]][m] - scores[sorted_front[k-1]][m]) / (scores[sorted_front[-1]][m] - scores[sorted_front[0]][m])
            distance[sorted_front[k]] += dist
    return distance

def binary_tournament_selection(population):
    fronts, scores = non_dominated_sort(population)
    rank = [None] * len(population)
    for i, front in enumerate(fronts):
        for idx in front:
            rank[idx] = i
    distances = [0] * len(population)
    for front in fronts:
        cd = crowding_distance(front, scores)
        for idx in front:
            distances[idx] = cd[idx]
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        if rank[i] < rank[j]:
            selected.append(population[i])
        elif rank[i] > rank[j]:
            selected.append(population[j])
        else:
            selected.append(population[i] if distances[i] > distances[j] else population[j])
    return selected

def differential_mutation(ind1, ind2, ind3, F=0.5):
    mutant_schedule = np.copy(ind1['schedule'])
    for idx in range(J):
        if random.random() < F:
            mutant_schedule[idx] = ind2['schedule'][idx]
    mutant_factory = np.copy(ind1['factory'])
    for idx in range(J):
        if random.random() < F:
            mutant_factory[idx] = ind3['factory'][idx]
    return {'schedule': mutant_schedule, 'factory': mutant_factory}

def crossover(parent, mutant, CR=0.9):
    child_schedule = np.copy(parent['schedule'])
    child_factory = np.copy(parent['factory'])
    for i in range(J):
        if random.random() < CR:
            child_schedule[i] = mutant['schedule'][i]
        if random.random() < CR:
            child_factory[i] = mutant['factory'][i]
    child_schedule = repair_permutation(child_schedule)
    return {'schedule': child_schedule, 'factory': child_factory}

def repair_permutation(schedule):
    missing = set(range(J)) - set(schedule)
    counts = {}
    new_schedule = []
    for i in schedule:
        counts[i] = counts.get(i, 0) + 1
    for i in schedule:
        if counts[i] > 1:
            counts[i] -= 1
            new_schedule.append(missing.pop() if missing else i)
        else:
            new_schedule.append(i)
    return np.array(new_schedule)

def mutate_individual(individual):
    new_ind = {'schedule': np.copy(individual['schedule']),
               'factory': np.copy(individual['factory'])}
    i, j = random.sample(range(J), 2)
    new_ind['schedule'][i], new_ind['schedule'][j] = new_ind['schedule'][j], new_ind['schedule'][i]
    idx = random.randint(0, J - 1)
    new_ind['factory'][idx] = random.randint(0, W - 1)
    return new_ind

def dominates(f1, f2):
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])

def local_search(individual, iterations=10):
    best = individual
    best_fitness = evaluate_individual(best)
    for _ in range(iterations):
        neighbor = mutate_individual(best)
        fitness = evaluate_individual(neighbor)
        if dominates(fitness, best_fitness):
            best = neighbor
            best_fitness = fitness
    return best

def sfl_dea():
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = (float('inf'), float('inf'))
    for _ in range(max_iterations):
        fitnesses = [evaluate_individual(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            if dominates(fit, best_fitness):
                best_solution = ind
                best_fitness = fit
        selected = binary_tournament_selection(population)
        offspring = []
        for i in range(len(selected)):
            idxs = random.sample(range(len(selected)), 3)
            ind1, ind2, ind3 = selected[idxs[0]], selected[idxs[1]], selected[idxs[2]]
            mutant = differential_mutation(ind1, ind2, ind3)
            child = crossover(selected[i], mutant)
            offspring.append(child)
        population = binary_tournament_selection(population + offspring)[:population_size]
        for i in range(len(population)):
            if random.random() < 0.1:
                population[i] = local_search(population[i])
    return best_solution, best_fitness

# -----------------------------
# Simulation using SimPy & Energy Computation from Timeline
# -----------------------------
timeline = []

def process_job(env, job, factory_id, op_machine_last_finish):
    for op in range(O):
        # Determine idle time if any on machine for current op in factory
        machine_ready = op_machine_last_finish.get((factory_id, op), 0)
        idle_time = max(0, env.now - machine_ready)
        if idle_time > SV:
            # Add turning on/off cost if idle time exceeds threshold
            env.energy_TEom += Eth_on_off
        env.energy_TEnlm += Unm[op, 0] * idle_time
        start_time = max(env.now, machine_ready)
        yield env.timeout(pt[job, op])
        finish_time = env.now
        timeline.append({'job': job, 'factory': factory_id, 'operation': op,
                         'start': start_time, 'finish': finish_time})
        op_machine_last_finish[(factory_id, op)] = finish_time
        # Transportation energy and delay for op < O-1
        if op < O - 1:
            yield env.timeout(th[op])
            env.energy_TEtr += Ulrt * th[op]
            env.energy_TEntr += Unrn * uh[op]
    return

def factory_process(env, factory_id, job_list, op_machine_last_finish):
    for job in job_list:
        yield env.process(process_job(env, job, factory_id, op_machine_last_finish))

def run_simulation(best_solution):
    env = simpy.Environment()
    env.energy_TElm = 0
    env.energy_TEnlm = 0
    env.energy_TEom = 0
    env.energy_TEtr = 0
    env.energy_TEntr = 0
    op_machine_last_finish = {}
    factory_jobs = {f: [] for f in range(W)}
    for job in best_solution['schedule']:
        f = best_solution['factory'][job]
        factory_jobs[f].append(job)
    for f in range(W):
        env.process(factory_process(env, f, factory_jobs[f], op_machine_last_finish))
    env.run()
    total_energy = (env.energy_TElm + env.energy_TEnlm +
                    env.energy_TEom + env.energy_TEtr + env.energy_TEntr)
    return timeline, factory_jobs, total_energy

def plot_gantt(timeline):
    colors = plt.cm.get_cmap('tab20', O)
    fig, ax = plt.subplots(figsize=(12, 8))
    yticks = []
    yticklabels = []
    height = 0.8
    job_events = {}
    for event in timeline:
        job = event['job']
        if job not in job_events:
            job_events[job] = []
        job_events[job].append(event)
    for idx, job in enumerate(sorted(job_events.keys())):
        events = sorted(job_events[job], key=lambda x: x['operation'])
        for ev in events:
            ax.broken_barh([(ev['start'], ev['finish'] - ev['start'])],
                           (idx - height/2, height),
                           facecolors=colors(ev['operation']))
        yticks.append(idx)
        yticklabels.append(f'Job {job}')
    ax.set_xlabel("Time")
    ax.set_ylabel("Jobs")
    ax.set_title("Gantt Chart of Job Processing")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    patches = [mpatches.Patch(color=colors(op), label=f'Op {op}') for op in range(O)]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    best_sol, best_fit = sfl_dea()
    print("Optimized Best Solution:")
    print("Schedule (order of jobs):", best_sol['schedule'])
    print("Factory assignment:", best_sol['factory'])
    print("Fitness (Makespan, Energy):", best_fit)
    
    sim_timeline, factory_jobs, sim_energy = run_simulation(best_sol)
    print("\nFactory Job Distribution:")
    for f in range(W):
        print(f"Factory {f}: Jobs {factory_jobs[f]}")
    print("\nSimulated Total Energy Consumption:", sim_energy)
    
    plot_gantt(sim_timeline)
