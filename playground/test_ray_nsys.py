import torch
import ray

ray.init()

@ray.remote(
num_gpus=1, 
runtime_env={ "nsight": "default"})
class RayActor:
    def run(self,):
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a * b
        print("Result on GPU:", c)

ray_actor = RayActor.remote()

# The Actor or Task process runs with :
# "nsys profile -t cuda,cudnn,cublas --cuda-memory-usage=True --cuda-graph-trace=graph ..."
# NOTE: the profiled result is located in "/tmp/ray/session_latest/logs/nsight/"
ray.get(ray_actor.run.remote())
