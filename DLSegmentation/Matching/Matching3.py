import MaskMatchingTorch
import time

t = MaskMatchingTorch.MaskMatchingTorch('../../Data/Sample1')
t.MaskLoader('../../Data/Sample1/masks', 'detectron2', 2)

start = time.time()
t.ComputeOffset(0)
end = time.time()

print(end - start)