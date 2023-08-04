import sys
from nseg.conf import NConf


dict_cfg = sys.stdin.read()

# print(dict_cfg)
# exit()
oc_cfg = NConf.create(dict_cfg)

yaml_cfg = NConf.to_yaml(oc_cfg)#, default_flow_style=None)
print(yaml_cfg)
