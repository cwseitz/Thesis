import tifffile
from pystackreg import StackReg
from pystackreg.util import to_uint16
from skimage.io import imsave
import json

prefixes = [
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___10',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___12',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___15',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___16',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___1',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___20',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___22',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___3',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___6',
'240117_SMT_silenceBRD4_646_2pm_1hour_L640_5mW_100ms___8'
]

with open('run_register.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    sr = StackReg(StackReg.TRANSLATION)
    path = config['datapath']+prefix
    stack = tifffile.imread(path+'.tif')
    out = sr.register_transform_stack(stack, reference='first', n_frames=10)
    path = config['datapath']+prefix
    imsave(path+'-regi.tif',to_uint16(out))
