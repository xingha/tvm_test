from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd
# import nnvm.compiler
# import nnvm.testing 2
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
from tvm.relay import testing
import cv2
import time
import numpy as np
import sklearn
from sklearn import preprocessing 

prefix,epoch = "/home/u260260/mntdisk/sets_face/face_models/r100/model",0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
opt_level = 3

shape_dict = {'data': (1, 3, 112, 112)}
# tag = tvm.target.Target("llvm -mcpu=skylake")
# tag = tvm.target.mali()
tag = "llvm -mcpu=skylake"
# target_host = "llvm"

def model_b():
   relay_mod, relay_params = relay.frontend.from_mxnet(symbol=sym,shape=shape_dict,dtype="float32",arg_params=arg_params, aux_params=aux_params)
   with tvm.transform.PassContext(opt_level=3):
      graph, tvm_mod, params = relay.build(relay_mod, tag, target_host=target_host,params=relay_params)
   tvm_mod.export_library("./tvm-model/deploy_lib.so")
   print('lib export succeefully')
   with open("./tvm-model/deploy_graph.json", "w") as fo:
      fo.write(graph)
   with open("./tvm-model/deploy_param.params", "wb") as fo:
      fo.write(relay.save_param_dict(params))


def model_c():
   shape_dict = {'data': (1, 3, 112, 112)}
   mod, relay_params = relay.frontend.from_mxnet(sym, shape_dict, arg_params=arg_params, aux_params=aux_params)
   ## we want a probability so add a softmax operator
   # func = mod["main"]
   # func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)
   # now compile the graph
   # target = "cuda"
   target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=armv8-linux-gnueabihf -mattr=+neon')
   # target = tvm.target.cuda(model="K620")
   with tvm.transform.PassContext(opt_level=3):
      graph, tvm_mod, params = relay.build(mod, target, target_host="llvm",params=relay_params)
      tvm_mod.export_library("./tvm-model/deploy_arm_lib.so")
   print('lib export succeefully')
   with open("./tvm-model/deploy_arm_graph.json", "w") as fo:
      fo.write(graph)
   with open("./tvm-model/deploy_arm_param.params", "wb") as fo:
      fo.write(relay.save_param_dict(params))


def runmodule(path):
   with open("./tvm-model/deploy_cuda_graph.json")as f1:
      grap = f1.read()
   with open("./tvm-model/deploy_cuda_param.params", "rb")as f2:
      params = bytearray(f2.read())
   lib = tvm.runtime.load_module("./tvm-model/deploy_cuda_lib.so")
   ctx = tvm.cpu()
   module = graph_runtime.create(grap, lib, ctx)
   module.load_params(params)
   im = cv2.imread(path)
   im = cv2.resize(im, (112,112))
   arrary = np.transpose(im,(2,0,1))
   arrary = np.expand_dims(arrary, axis=0)
   t1 = time.time()
   module.run(data=arrary)
   module.run(data=arrary)
   
   # print(module.get_output(0))
   print("recognition time:", time.time() - t1)
   for i in range(module.get_num_outputs()):
      out = module.get_output(i).asnumpy()
      embedding = preprocessing.normalize(out).flatten()
      # print(embedding)
      print('[%d]'%i, type(embedding))
      print('[%d]'%i, embedding.shape)
git push -u origin main

class Performance:
   def __init__(self, path):
      self.filepath = path
      with open('./tvm-model/deploy_graph.json')as fgraph:
         graph = fgraph.read()
      with open('./tvm-model/deploy_param.params', 'rb')as fparam:
         params = bytearray(fparam.read())
      lib = tvm.runtime.load_module('./tvm-model/deploy_lib.so')
      ctx = tvm.cpu()
      module = graph_runtime.create(graph, lib, ctx)
      module.load_params(params)
      self.module = module

   def performance(self):
      numb_t = 0
      with open(self.filepath) as f:
         lines = f.readlines()
      for line in lines:
         imgfile1, imgfile2, flag = line.strip().split(' ')
         print(imgfile1.split('/')[-1], imgfile2.split('/')[-1])
         ret = self.matchfeature(imgfile1, imgfile2)
         if ret == int(flag):
            numb_t +=1
      count_all = len(lines)
      print("[COUNT]{} [RATE]{} [TRUE]{}".format(count_all, numb_t/count_all, numb_t))

   def matchfeature(self, f1=None, f2=None):
      data1 = processimg(f1)
      data2 = processimg(f2)
      self.module.run(data=data1)
      out1 = self.module.get_output(0).asnumpy()
      out1 = preprocessing.normalize(out1).flatten()
      print("<1out shape>",out1.shape)
      self.module.run(data=data2)
      out2 = self.module.get_output(0).asnumpy()
      out2 = preprocessing.normalize(out2).flatten()
      print("<2out shape>",out2.shape)
      cosin = np.dot(out1, out2.T)
      print("similarity:", cosin)
      if cosin>0.6:
         return 1
      return 0


def processimg(imgpath):
   im = cv2.imread(imgpath)
   w,h,c = im.shape
   assert w == 112 and h == 112
   im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
   arrary = np.transpose(im, (2,0,1))
   data = np.expand_dims(arrary, 0)
   return data


def mxnet_get(mxnetpath, layer):
   sym,arg_params,aux_params = mx.model.load_checkpoint(mxnetpath,0)
   all_layers = sym.get_internals()
   sym = all_layers[layer+'_output']
   model = mx.mod.Module(sym, label_names=None, context=mx.cpu())
   model.bind(data_shapes=[("data",(1,3,112,112))])
   model.set_params(arg_params=arg_params, aux_params=aux_params)
   return model

class MxnetModel:
   def __init__(self,modelpath, filepath):
      self.model = mxnet_get(modelpath, "fc1")
      self.lst = filepath

   def get_feature(self, im_array):
      input_db = mx.nd.array(im_array,ctx=mx.cpu())
      print(type(input_db))
      db = mx.io.DataBatch(data=(input_db,))
      self.model.forward(db, is_train=False)
      embeddings = self.model.get_outputs()[0].asnumpy()
      embeddings = preprocessing.normalize(embeddings).flatten()
      return embeddings

   def performance(self):
      with open(self.lst) as f:
         lines = f.readlines()
      numb_t = 0
      for line in lines:
         file1, file2, flag = line.strip().split(' ')
         im1 = processimg(file1)
         im2 = processimg(file2)
         st = time.time()
         embed1 = self.get_feature(im1)
         embed2 = self.get_feature(im2)
         et = time.time()
         cosin = np.dot(embed1, embed2.T)
         print("[get time]",et-st,"<similarity>",cosin)
         if cosin>0.6:
            ret = 1
         else:
            ret = 0
         if ret == int(flag):
            numb_t += 1
      count_all = len(lines)
      print("[COUNT]{} [RATE]{} [TRUE]{}".format(count_all, numb_t/count_all, numb_t))


if __name__ == "__main__":
   # model_b()
   model_c()
   # runmodule("t2.jpg")
   # path ="/home/u260260/mntdisk/Cspace/captureframe/conver-tvm/conver1/multi/multi.lst"
   # pe = Performance(path)
   # pe.performance()
   # md = MxnetModel(prefix, path)
   # md.performance()
