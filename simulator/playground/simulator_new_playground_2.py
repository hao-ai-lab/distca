# %%
import numpy as np
import pyomo.environ as pyo

# If you already have your simulator module:
# from simulator.simulator import get_batch_attn_time

# Otherwise, define a stub for testing:
def get_batch_attn_time(batch, tp, cp, do_sum=False):
    # Replace with your real timing function!
    # Here we just return random latencies for illustration.
    # Returns a numpy array of length batch.shape[0].
    return np.random.rand(batch.shape[0])

# %%
# Example “batch” of data indices; replace with your real batch array.
batch = np.arange(20)

# total devices available, and max workers you want to consider
num_total_devices = 8
num_worker_max    = 4


# %%

# 4.1 parallel_plan = list of (tp,cp) pairs
parallel_plan = []
for tp in (1,2,4,8):
    for cp in (1,2,4,8):
        if tp*cp <= min(8, num_total_devices):
            parallel_plan.append((tp,cp))

# add the idle option
parallel_plan.append((0,0))

# 4.2 resource usage per plan
resource = np.array([tp * cp for tp, cp in parallel_plan])

# 4.3 latency[j,k] = time of plan j on data item k
num_sols = len(parallel_plan)
num_data = batch.shape[0]
latency = np.zeros((num_sols, num_data))
for j, (tp,cp) in enumerate(parallel_plan):
    latency[j] = get_batch_attn_time(batch, tp, cp, do_sum=False)


# %%
model = pyo.ConcreteModel()

# index sets
model.I = pyo.RangeSet(0, num_worker_max-1)
model.J = pyo.RangeSet(0, num_sols-1)
model.K = pyo.RangeSet(0, num_data-1)

# decision vars
model.x       = pyo.Var(model.I, model.J, domain=pyo.Binary)
model.y       = pyo.Var(model.K, model.I, domain=pyo.Binary)
model.lat_max = pyo.Var(domain=pyo.NonNegativeReals)

# 5.1 each worker picks exactly one plan
def one_plan(m,i):
    return sum(m.x[i,j] for j in m.J) == 1
model.one_plan = pyo.Constraint(model.I, rule=one_plan)

# 5.2 each data item assigned once
def one_assign(m,k):
    return sum(m.y[k,i] for i in m.I) == 1
model.one_assign = pyo.Constraint(model.K, rule=one_assign)

# 5.3 resource limit
def res_limit(m):
    return sum(m.x[i,j] * float(resource[j]) 
               for i in m.I for j in m.J) <= num_total_devices
model.res_limit = pyo.Constraint(rule=res_limit)

# 5.4 latency bound for each worker
def lat_bound(m,i):
    return sum(m.x[i,j] * m.y[k,i] * float(latency[j,k])
               for j in m.J for k in m.K) <= m.lat_max
model.lat_bound = pyo.Constraint(model.I, rule=lat_bound)

# 5.5 objective: minimize max latency
model.obj = pyo.Objective(expr=model.lat_max, sense=pyo.minimize)

# %%
solver = pyo.SolverFactory('couenne')
result = solver.solve(model, tee=True)

print("Solver status:", result.solver.status)
print("Termination:",   result.solver.termination_condition)
print("Optimal max‑latency:", pyo.value(model.lat_max))


# %%

results = []
for i in model.I:
    # find which plan j is chosen
    chosen_j = next(j for j in model.J if pyo.value(model.x[i,j]) > 0.5)
    tp, cp = parallel_plan[chosen_j]
    # collect data items assigned to worker i
    assigned = [k for k in model.K if pyo.value(model.y[k,i]) > 0.5]
    if assigned:
        # slice out those batch entries
        results.append(( batch[assigned], tp, cp ))

# display
for idx,(data_slice,tp,cp) in enumerate(results):
    print(f"Worker {idx:>2}: TP={tp}, CP={cp}, data indices={list(data_slice)}")