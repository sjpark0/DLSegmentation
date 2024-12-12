import MaskMatching

t = MaskMatching.MaskMatching('../../Data/Sample1')
t.MaskLoader('../../Data/Sample1/masks', 'detectron2', 2)
t.ComputeOffset(0)