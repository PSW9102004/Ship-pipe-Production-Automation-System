# Reinforcement Learning-Enhanced Scheduling

A hybrid scheduling system combining multi-objective SFL-DEA optimization and reinforcement learning for resilient, energy-efficient, and adaptive production in intelligent manufacturing.

## Features

* **Multi-Objective Optimization**: Minimizes makespan and total energy consumption simultaneously using an Integer Programming–based SFL-DEA metaheuristic.
* **Reinforcement Learning Integration**: A Q-learning agent adapts schedules in real time to handle machine failures and dynamic job arrivals.
* **Detailed Energy Model**: Accounts for load, no-load, and on/off energy consumption, plus robot transport energy.
* **Good-Point Initialization & Adaptive DE**: Improves convergence speed and solution quality through specialized population seeding and self-adaptive mutation/crossover parameters.

## Repository Structure

```
├── CIMS_file.py       # Main implementation combining SFL-DEA and RL modules
├── data/              # Example input datasets (job definitions, machine specs)
├── results/           # Generated Gantt charts, log files, Q-table snapshots
├── utils/             # Helper modules (encoding, decoding, energy calculations)
├── README.md          # Project overview and setup instructions
└── LICENSE            # MIT License
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rl-sfl-de-scheduler.git
   cd rl-sfl-de-scheduler
   ```
2. Create a Python virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

* **Train RL Agent**:

  ```bash
  python train_agent.py --episodes 1000
  ```

* **Run Scheduler**:

  ```bash
  python CIMS_file.py
  ```

  This will:

  * Initialize population with good-point set.
  * Execute SFL-DEA for multi-objective scheduling.
  * Invoke RL agent for dynamic adjustments.
  * Output Gantt chart and performance metrics.

## Configuration

Edit `config.yaml` to adjust:

* Job and operation counts (J, O).
* Factory and machine settings (W, Wp).
* Energy parameters (E\_onoff, U\_l, U\_nm, robot energy).
* SFL-DEA hyperparameters (NP, maxI, uf, uCR, beta, mu).
* RL agent settings (alpha, gamma, epsilon).

## Results

Results (Pareto front, Gantt charts) are saved in `/results`. Open the HTML or image files to visualize schedules and compare performance under different scenarios.

## References

* Xuan et al., *Ship pipe production optimization...* Expert Systems With Applications, 2025.
* Sutton & Barto, *Reinforcement Learning: An Introduction*, 2018.
* Pinedo, *Scheduling: Theory, Algorithms, and Systems*, 2016.

---

*Made with ❤️ for intelligent manufacturing research and development.*
