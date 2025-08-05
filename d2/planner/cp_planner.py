# Use Knapsack problem to solve CP MLP layout and dispatch.
# Target:
# P1: Number of tokens balanced on each rank
# P2: Attention flops balanced on each rank
# P3: Communication cost less is better


import pulp as pl
import math

# --------------------------------------------------
# 1. Data Definition (Unchanged)
# --------------------------------------------------

n, b = 15, 8 # sequences, ranks
K = 1024

c = {j: 32*K for j in range(b)}
# Mass of each item


m = {0: 64 *K, 1: 32 *K, 2: 32 * K, 3: 16*K, 4: 16*K, 5: 16*K, 6: 16*K, 7: 8*K, 8: 8*K, 9: 8*K, 10: 8*K, 11: 8*K, 12: 8*K, 13: 8*K, 14: 8*K}
# Possible ways to split each item 'i' into 'a' chunks, where 'a' is a power of 2
A = {i: [2**j for j in range(int(math.log2(b)) + 1)] for i in range(n)}

# --------------------------------------------------
# 2. Priority Weights (Unchanged)
# --------------------------------------------------
lambda_volume_balance = 4
lambda_cost_balance = 2
lambda_manual_cost = 1

# --------------------------------------------------
# 3. Create Model and Variables
# --------------------------------------------------
model = pl.LpProblem("Warehouse_Final_Logic_Simplified_MIP", pl.LpMinimize)

# Decision Variable: Choose Splitting Method (Unchanged)
# y[i, a] = 1 if item i is split into 'a' chunks, 0 otherwise
y = pl.LpVariable.dicts("y", ((i, a) for i in range(n) for a in A[i]), cat="Binary")

# --- Core Change: x is now a binary variable, representing placement, not quantity ---
# x[i, a, j] = 1 if a chunk of item i (split into 'a' parts) is placed in warehouse j
x = pl.LpVariable.dicts("x_placement", ((i, a, j) for i in range(n) for a in A[i] for j in range(b)), cat="Binary")

# --- Simplification: p variable is no longer needed ---

# P1 Objective Variable (Unchanged)
# Represents the maximum deviation from the average volume per warehouse
z_volume_dev = pl.LpVariable("z_volume_deviation", lowBound=0)

# P2 Objective Variables (Unchanged)
# Represent the max and min storage cost across all warehouses
C_max = pl.LpVariable("C_max", lowBound=0)
C_min = pl.LpVariable("C_min", lowBound=0)

# --------------------------------------------------
# 4. Constraints (More Concise)
# --------------------------------------------------
# 4.1 Choose Splitting Method (Unchanged)
# Each item must be split using exactly one method.
for i in range(n):
    model += pl.lpSum(y[i, a] for a in A[i]) == 1, f"Choose_One_Method_{i}"

# 4.2 Chunk Allocation Constraint (Unchanged, but now with a stronger meaning)
# If item i is split into 'a' chunks, then exactly 'a' chunks must be placed in warehouses.
# - Forces 'a' chunks to be placed in 'a' different warehouses
for i in range(n):
    for a in A[i]:
        model += pl.lpSum(x[i, a, j] for j in range(b)) == a * y[i, a], f"Distribute_to_a_warehouses_{i}_{a}"

# 4.3 Capacity Limit (Unchanged)
# The total mass placed in a warehouse cannot exceed its capacity.
for j in range(b):
    model += pl.lpSum(m[i] / a * x[i, a, j] for i in range(n) for a in A[i]) <= c[j], f"Capacity_{j}"

# 4.4 Volume Balancing (Unchanged)
# Defines the volume deviation variable z_volume_dev.
average_volume = sum(m.values()) / b
for j in range(b):
    used_volume = pl.lpSum(m[i] / a * x[i, a, j] for i in range(n) for a in A[i])
    model += used_volume - average_volume <= z_volume_dev, f"Volume_Dev_Upper_{j}"
    model += used_volume - average_volume >= -z_volume_dev, f"Volume_Dev_Lower_{j}"

# --- Simplification: No need for constraints linking p and x ---

# 4.5 Define Cost Range (Using x instead of p)
# Defines C_max and C_min based on the storage costs in each warehouse.
warehouse_storage_costs = [
    pl.lpSum((m[i]**2 / a) * x[i, a, j] for i in range(n) for a in A[i])
    for j in range(b)
]
for j in range(b):
    model += warehouse_storage_costs[j] <= C_max, f"Cost_Range_Upper_{j}"
    model += warehouse_storage_costs[j] >= C_min, f"Cost_Range_Lower_{j}"

# --------------------------------------------------
# 5. Final Objective Function (Unchanged)
# --------------------------------------------------
# P3: Manual cost associated with splitting items.
manual_cost = pl.lpSum(a * (a - 1) * m[i] * y[i, a] for i in range(n) for a in A[i])
# The range between the highest and lowest warehouse storage cost.
cost_range = C_max - C_min
# The combined, weighted objective function.
model += (
    lambda_volume_balance * z_volume_dev +
    lambda_cost_balance * cost_range +
    lambda_manual_cost * manual_cost
), "Hierarchical_Final_Objective"

# --------------------------------------------------
# 6. Solve (Unchanged)
# --------------------------------------------------
model.solve(pl.PULP_CBC_CMD(msg=True))

# --------------------------------------------------
# 7. Print Results (Unchanged, logic still applies)
# --------------------------------------------------
print("\n" + "="*60)
print(" " * 20 + "Optimization Solution Report")
print("="*60)

if model.status != pl.LpStatusOptimal:
    print("Solution failed, could not find an optimal solution. Status:", pl.LpStatus[model.status])
else:
    # --- Overall Summary ---
    print("\n[1. Overall Summary]")
    print(f"  - Solution Status: {pl.LpStatus[model.status]}")
    manual_cost_val = pl.value(manual_cost)
    print(f"  - Total Manual Cost: {manual_cost_val:.2f}")
    cost_range_val = pl.value(cost_range)
    vol_dev_val = pl.value(z_volume_dev)
    print("\n  - Objective Function Details:")
    print(f"    - Total Objective Value: {pl.value(model.objective):.2f}")
    print(f"    - P1 (Volume Balance) Loss: {vol_dev_val:.4f} (Max Deviation)")
    print(f"    - P2 (Cost Balance) Loss: {cost_range_val:.4f} (Cost Range Max-Min)")
    print(f"    - P3 (Manual Cost) Loss: {manual_cost_val:.2f}")

    # --- Display Allocation Plan by Item ---
    print("\n[2. Allocation Plan Details by Item]")
    for i in range(n):
        chosen_a = -1
        for a in A[i]:
            if y[i, a].value() > 0.5:
                chosen_a = a
                break
        
        print(f"\n* Item {i} (Original Mass {m[i]}):")
        print(f"  - Decision: Chosen to be split into {chosen_a} parts.")
        print(f"  - Destination:")
        
        for j in range(b):
            if x[i, chosen_a, j].value() > 0.5:
                print(f"    - 1 chunk -> Warehouse {j}")

    # --- Display Content List by Warehouse ---
    print("\n[3. Content List by Warehouse]")
    final_costs = [pl.value(c) for c in warehouse_storage_costs]
    for j in range(b):
        print(f"\n* Warehouse {j}:")
        total_mass_in_wh = 0
        pieces_info = []
        for i in range(n):
            for a in A[i]:
                if x[i, a, j].value() > 0.5:
                    mass = m[i]/a
                    total_mass_in_wh += mass
                    cost = m[i]*m[i]/a
                    pieces_info.append(f"    - Source: Item {i} (Original Mass {m[i]}), Mass: {mass:.2f}, Cost: {cost:.2f}")
        print(f"  - Total Volume Used: {total_mass_in_wh:.2f} / {c[j]}")
        print(f"  - Total Storage Cost: {final_costs[j]:.2f}")
        print(f"  - Total Chunks Stored: {len(pieces_info)}.")
        if not pieces_info:
            print("  - Warehouse is empty.")
        else:
            print("  - Content Details:")
            for info in pieces_info:
                print(info)

print("\n" + "="*60)