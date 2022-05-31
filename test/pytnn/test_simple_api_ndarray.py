import pytnn
import numpy as np

input=np.ones((6,3,32,320), np.float32, 'F')

# torchscript 
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[ {"min": [1,3,224,224], "max": [1,3,224,224]} ], "network_type": pytnn.NETWORK_TYPE_TNNTORCH, "device_type": pytnn.DEVICE_CUDA})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[ {"min": [1,3,224,224], "max": [1,3,224,224]} ], "network_type": "tnntorch", "device_type": "cuda"})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[ {"min": [1,3,224,224], "max": [1,3,224,224]} ]})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[[1,3,224,224]]})
#module=pytnn.load("../../model/SqueezeNet/squeezenet_v1.1.ts", {"input_shapes":[(1,3,224,224)]})
#output=module.forward(input)
#print(output[0])

# tnnproto
module=pytnn.load("/data/yinru/tmp/TNN/test/pytnn/tmp1")
#module=pytnn.optimize("/data/yinru/ocrOnnx/data/models/ocrModel/inference/rec_onnx",1,0,input_shapes=['x:array.float(6*3*32*320)'], save_path="/data/yinru/tmp/TNN/test/pytnn/tmp1")
output1=module.forward(input)
print(output1[0])

