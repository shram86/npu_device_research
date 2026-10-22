"""PaDiM Experiment B: normalized image -> ResNet features -> FP32 Gaussian scores."""
import ctypes
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SHAPES = ((64,64,64),(128,32,32),(256,16,16))
MEAN = np.array([.485,.456,.406],np.float32)
STD = np.array([.229,.224,.225],np.float32)


def crop_position(position=None):
    position={'x':.5,'y':.5} if position is None else position
    if not isinstance(position,dict) or set(position)!={'x','y'}:
        raise ValueError('crop_position requires x and y in [0,1]')
    if any(isinstance(position[k],bool) or not isinstance(position[k],(int,float)) or
           not np.isfinite(position[k]) or not 0<=position[k]<=1 for k in ('x','y')):
        raise ValueError('crop_position requires finite x and y in [0,1]')
    return dict(x=float(position['x']),y=float(position['y']))


def resize_geometry(width,height,position=None):
    position=crop_position(position)
    if min(width,height) < 1:
        raise ValueError('Invalid image dimensions')
    if width <= height:
        nw,nh = 256,int(256*height/width)
    else:
        nw,nh = int(256*width/height),256
    return nw,nh,int((nw-256)*position['x']),int((nh-256)*position['y'])


def view_geometry(width,height,position=None):
    nw,nh,left,top = resize_geometry(width,height,position)
    x,y = round(left*width/nw),round(top*height/nh)
    return dict(x=x,y=y,width=min(width-x,round(256*width/nw)),height=min(height-y,round(256*height/nh)))


class Preprocessor:
    """Select reference PIL or fast native position-aware crop preprocessing."""
    def __init__(self,mode='pil',library=None,position=None):
        self.position=crop_position(position)
        if mode not in ('pil','neon'):
            raise ValueError('Preprocessing must be pil or neon')
        self.mode=mode
        self.library=None
        if mode=='neon':
            if library is None:
                raise ValueError('Native preprocessing requires libpadim_camera.so')
            self.library=ctypes.CDLL(str(library)) if isinstance(library,(str,Path)) else library
            try:self.native=self.library.padim_preprocess_neon_position
            except AttributeError as error:
                raise RuntimeError('Missing padim_preprocess_neon_position; rebuild libpadim_camera.so') from error
            self.native.argtypes=[ctypes.POINTER(ctypes.c_uint8)]+[ctypes.c_int]*3+[ctypes.c_double]*2+[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
            self.native.restype=ctypes.c_int
            self.output=np.empty((1,3,256,256),np.float32)
            self.output_pointer=self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def __call__(self,frame,roi=None):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected uint8 BGR camera frame')
        height,width = frame.shape[:2]
        if self.mode=='neon':
            stride=frame.strides[0]
            if frame.strides[1:]!=(3,1) or stride<width*3:
                raise ValueError('Native preprocessing requires packed BGR pixels with positive row stride')
            if max(width,height,stride)>2147483647:
                raise ValueError('Image dimensions exceed native int32 limits')
            status=self.native(frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),width,height,stride,self.position['x'],self.position['y'],
                               self.output_pointer,self.output.size)
            if status:
                raise ValueError('Native preprocessing rejected image dimensions or buffers')
            return self.output
        nw,nh,left,top = resize_geometry(width,height,self.position)
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image = image.resize((nw,nh),Image.Resampling.BILINEAR).crop((left,top,left+256,top+256))
        rgb = np.asarray(image,dtype=np.float32)/np.float32(255)
        return np.ascontiguousarray(((rgb-MEAN)/STD).transpose(2,0,1)[None])


def resize_bilinear_chw(value,size=64):
    """Half-pixel centers / align_corners=False; no PyTorch dependency on the board."""
    _,height,width = value.shape
    if (height,width)==(size,size):
        return value
    x = np.clip((np.arange(size,dtype=np.float32)+.5)*np.float32(width/size)-.5,0,width-1)
    y = np.clip((np.arange(size,dtype=np.float32)+.5)*np.float32(height/size)-.5,0,height-1)
    x0,y0 = x.astype(np.int64),y.astype(np.int64)
    x1,y1 = np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
    wx,wy = (x-x0).astype(np.float32)[None,None,:],(y-y0).astype(np.float32)[None,:,None]
    top = value[:,y0[:,None],x0[None,:]]*(1-wx)+value[:,y0[:,None],x1[None,:]]*wx
    bottom = value[:,y1[:,None],x0[None,:]]*(1-wx)+value[:,y1[:,None],x1[None,:]]*wx
    return top*(1-wy)+bottom*wy


class PadimScorer:
    def __init__(self,path,load_precision=True):
        with np.load(path,allow_pickle=False) as stats:
            self.mean = np.ascontiguousarray(stats['mean'],dtype=np.float32)
            self.precision = np.ascontiguousarray(stats['precision'],dtype=np.float32) if load_precision else None
            self.indices = np.array(stats['indices'],dtype=np.int64)
            if 'image_size' in stats and int(stats['image_size']) != 256:
                raise ValueError('Statistics require a different image size')
        if self.mean.shape != (4096,100) or (load_precision and self.precision.shape != (4096,100,100)):
            raise ValueError('Expected FP32 PaDiM mean [4096,100] and precision [4096,100,100]')
        if (self.indices.shape != (100,) or len(np.unique(self.indices)) != 100 or
                self.indices.min()<0 or self.indices.max()>=448):
            raise ValueError('Invalid PaDiM channel selection')
        if not np.isfinite(self.mean).all() or (load_precision and not np.isfinite(self.precision).all()):
            raise ValueError('Nonfinite PaDiM statistics')
        self.selected = np.empty((100,64,64),np.float32)
        self.native_distance=None
        self.delta=np.empty((4096,100),np.float32)
        self.distance=np.empty(4096,np.float32)

    def bind_native(self,library):
        # Older libraries retain the reference NumPy path until rebuilt.
        function=getattr(library,'padim_mahalanobis',None)
        if function is None:return
        fp=ctypes.POINTER(ctypes.c_float)
        function.argtypes=[fp,fp,fp,ctypes.c_size_t,ctypes.c_size_t]
        function.restype=ctypes.c_int
        self.native_distance=function

    def select_features(self,features):
        if len(features)!=3:
            raise ValueError('Expected three ResNet feature tensors')
        offset=0
        for value,shape in zip(features,SHAPES):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid feature tensor, expected {shape}')
            positions = np.flatnonzero((self.indices>=offset)&(self.indices<offset+shape[0]))
            # Selecting before independent spatial resize is equivalent to concat-then-select.
            if positions.size:
                channels = self.indices[positions]-offset
                self.selected[positions] = resize_bilinear_chw(value[channels])
            offset+=shape[0]
        return self.selected

    def __call__(self,features):
        return self.score_selected(self.select_features(features))

    def score_selected(self,selected):
        if selected.shape!=(100,64,64) or selected.dtype!=np.float32 or not np.isfinite(selected).all():
            raise ValueError('Expected finite FP32 selected features [100,64,64]')
        x = selected.transpose(1,2,0).reshape(4096,100)
        np.subtract(x,self.mean,out=self.delta)
        if self.native_distance is None:
            d2 = np.einsum('pc,pcd,pd->p',self.delta,self.precision,self.delta,optimize=True)
        else:
            fp=ctypes.POINTER(ctypes.c_float)
            if self.native_distance(self.delta.ctypes.data_as(fp),self.precision.ctypes.data_as(fp),
                                    self.distance.ctypes.data_as(fp),4096,100):
                raise RuntimeError('Native PaDiM distance failed')
            d2=self.distance
        anomaly_map = gaussian_filter(np.sqrt(np.maximum(d2,0)).reshape(64,64),sigma=2)
        if not np.isfinite(anomaly_map).all():
            raise ValueError('Nonfinite PaDiM distance map')
        return anomaly_map,float(anomaly_map.max())


class NPUScorer(PadimScorer):
    """NB embeds mean/precision; host sends selected features, not deltas."""
    def __init__(self,config,library,owner):
        super().__init__(config['statistics'],load_precision=False)
        meta=json.loads(Path(config['score_metadata']).read_text())
        for key,shape in [('Inputs',[1,100,64,64]),('Outputs',[1,1,64,64])]:
            entries=list(meta[key].values())
            if len(entries)!=1 or entries[0]['shape']!=shape or entries[0]['format'].lower()!='nchw' or entries[0]['dtype'].lower()!='float16':
                raise ValueError('Score metadata must describe FP16 NCHW features and squared distance')
        if not Path(config['score_model']).is_file():raise FileNotFoundError(config['score_model'])
        self.lib=library;self.handle=None
        try:
            self.lib.padim_score_open.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t]
            self.lib.padim_score_open.restype=ctypes.c_void_p
            self.lib.padim_score_close.argtypes=[ctypes.c_void_p];self.lib.padim_score_close.restype=None
            hp=ctypes.POINTER(ctypes.c_uint16)
            self.lib.padim_score_run.argtypes=[ctypes.c_void_p,hp,ctypes.c_size_t,hp,ctypes.c_size_t,
                                              ctypes.POINTER(ctypes.c_double),ctypes.c_char_p,ctypes.c_size_t]
            self.lib.padim_score_run.restype=ctypes.c_int
        except AttributeError as error:raise RuntimeError('Rebuild libpadim_camera.so for NPU scoring') from error
        self.error=ctypes.create_string_buffer(1024)
        self.handle=self.lib.padim_score_open(owner,str(config['score_model']).encode(),self.error,len(self.error))
        if not self.handle:raise RuntimeError(self.error.value.decode(errors='replace'))
        self.input_half=np.empty((100,64,64),np.float16)
        self.output_half=np.empty((64,64),np.float16)
        self.output_float=np.empty((64,64),np.float32)
        self.timings=(ctypes.c_double*3)();self.last_timings={}

    def __call__(self,features):
        return self.score_selected(self.select_features(features))

    def score_selected(self,selected):
        started=time.monotonic()
        if selected.shape!=(100,64,64) or selected.dtype!=np.float32 or not np.isfinite(selected).all():
            raise ValueError('Expected finite FP32 selected features [100,64,64]')
        np.copyto(self.input_half,selected,casting='unsafe')
        if not np.isfinite(self.input_half).all():raise ValueError('Score input overflows FP16')
        ready=time.monotonic();hp=ctypes.POINTER(ctypes.c_uint16)
        if self.lib.padim_score_run(self.handle,self.input_half.ctypes.data_as(hp),self.input_half.size,
                self.output_half.ctypes.data_as(hp),self.output_half.size,self.timings,self.error,len(self.error)):
            raise RuntimeError(self.error.value.decode(errors='replace'))
        done=time.monotonic()
        np.copyto(self.output_float,self.output_half)
        if not np.isfinite(self.output_float).all():raise ValueError('Nonfinite NPU squared distance (FP16 overflow)')
        np.maximum(self.output_float,0,out=self.output_float)
        np.sqrt(self.output_float,out=self.output_float)
        anomaly_map=gaussian_filter(self.output_float,sigma=2)
        self.last_timings=dict(feature_select_ms=(ready-started)*1000,
                              score_input_ms=self.timings[0],score_network_ms=self.timings[1],
                              score_output_ms=self.timings[2],score_finish_ms=(time.monotonic()-done)*1000)
        return anomaly_map,float(anomaly_map.max())

    def close(self):
        if self.handle:self.lib.padim_score_close(self.handle);self.handle=None


def read_metadata(path):
    meta = json.loads(Path(path).read_text())
    inputs = list(meta['Inputs'].values())
    if len(inputs)!=1 or inputs[0]['shape'] != [1,3,256,256] or inputs[0]['format'].lower()!='nchw':
        raise ValueError('Expected NCHW batch-1 256x256 metadata')
    q = inputs[0]['quantize']
    if q['qtype'].lower() not in ('i8','int8'):
        raise ValueError('Experiment B expects signed INT8 input')
    scale,zero=float(np.float32(q['scale'])),int(q['zero_point'])
    if not np.isfinite(scale) or scale<=0 or not -128<=zero<=127:
        raise ValueError('Invalid input quantization')
    outputs={}
    for entry in meta['Outputs'].values():
        shape=tuple(entry['shape'][1:])
        if entry['shape'][0]!=1 or entry['format'].lower()!='nchw' or shape not in SHAPES or shape in outputs:
            raise ValueError('Expected unique batch-1 NCHW layer1/layer2/layer3 metadata')
        outputs[shape]=entry
    if set(outputs)!=set(SHAPES):
        raise ValueError('Missing feature output metadata')
    return scale,zero,outputs


class NBBackend:
    def __init__(self,config):
        self.handle=None
        for name in ('model','library','metadata','statistics'):
            if not Path(config[name]).is_file():
                raise FileNotFoundError(f"Missing {name}: {config[name]}. Select the tested NB and its matching metadata; see README.")
        self.scale,self.zero,outputs=read_metadata(config['metadata'])
        self.scorer=None
        self.lib=ctypes.CDLL(config['library'])
        fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
        self.lib.padim_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_size_t]
        self.lib.padim_open.restype=ctypes.c_void_p
        self.lib.padim_close.argtypes=[ctypes.c_void_p];self.lib.padim_close.restype=None
        self.lib.padim_run.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int8),ctypes.c_size_t,fp,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t]
        self.lib.padim_run.restype=ctypes.c_int
        self.lib.padim_io_info.argtypes=[ctypes.c_void_p,ctypes.c_int,dp,ctypes.c_size_t]
        self.lib.padim_io_info.restype=ctypes.c_int
        self.lib.padim_get_timings.argtypes=[ctypes.c_void_p,dp,ctypes.c_size_t]
        self.lib.padim_get_timings.restype=ctypes.c_int
        self.error=ctypes.create_string_buffer(1024)
        self.handle=self.lib.padim_open(config['model'].encode(),self.error,len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode(errors='replace'))
        try:
            if config.get('scoring','cpu')=='npu':
                self.scorer=NPUScorer(config,self.lib,self.handle)
            else:
                self.scorer=PadimScorer(config['statistics'])
                self.scorer.bind_native(self.lib)
            self.selected_readback=hasattr(self.scorer,'score_selected')
            if self.selected_readback:
                try:
                    self.lib.padim_set_channels.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
                    self.lib.padim_set_channels.restype=ctypes.c_int
                    self.lib.padim_run_selected.argtypes=self.lib.padim_run.argtypes
                    self.lib.padim_run_selected.restype=ctypes.c_int
                except AttributeError as error:
                    raise RuntimeError('Rebuild libpadim_camera.so for selected-channel readback') from error
                indices=np.ascontiguousarray(self.scorer.indices,dtype=np.int32)
                if self.lib.padim_set_channels(self.handle,indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),indices.size):
                    raise ValueError('Invalid native PaDiM selected channels')
            info=(ctypes.c_double*3)()
            for index,expected in [(-1,dict(quantize=dict(scale=self.scale,zero_point=self.zero)))]+list(enumerate(outputs[shape] for shape in SHAPES)):
                if self.lib.padim_io_info(self.handle,index,info,3):
                    raise RuntimeError('Cannot query runtime quantization')
                floating,scale,zero=list(info)
                if floating:
                    if index<0:raise ValueError('Runtime input must be INT8')
                    continue # Floating output is already dequantized: do not rescale it.
                q=expected['quantize']
                if not np.isclose(scale,q['scale'],rtol=1e-6,atol=1e-10) or zero!=q['zero_point']:
                    raise ValueError(f'Runtime/metadata quantization mismatch at layer {index}; use matching metadata')
        except Exception:
            self.close();raise
        self.quantized=np.empty((1,3,256,256),np.int8)
        self.qwork=np.empty((1,3,256,256),np.float32)
        count=409600 if self.selected_readback else sum(int(np.prod(shape)) for shape in SHAPES)
        self.feature_buffer=np.empty(count,np.float32)
        self.selected_features=self.feature_buffer.reshape(100,64,64) if self.selected_readback else None
        self.run_native=self.lib.padim_run_selected if self.selected_readback else self.lib.padim_run
        self.features=[];start=0
        for shape in (() if self.selected_readback else SHAPES):
            count=int(np.prod(shape));self.features.append(self.feature_buffer[start:start+count].reshape(shape));start+=count
        self.timings=(ctypes.c_double*3)();self.last_timings={}

    def run(self,image):
        if image.shape!=(1,3,256,256) or image.dtype!=np.float32 or not np.isfinite(image).all():
            raise ValueError('Expected finite normalized float32 NCHW image')
        started=time.monotonic()
        np.divide(image,self.scale,out=self.qwork);self.qwork+=self.zero
        np.rint(self.qwork,out=self.qwork);np.clip(self.qwork,-128,127,out=self.qwork)
        np.copyto(self.quantized,self.qwork,casting='unsafe')
        ready=time.monotonic()
        if self.run_native(self.handle,self.quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),self.quantized.size,
                              self.feature_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),self.feature_buffer.size,self.error,len(self.error)):
            raise RuntimeError(self.error.value.decode(errors='replace'))
        native_done=time.monotonic()
        result=self.scorer.score_selected(self.selected_features) if self.selected_readback else self.scorer(self.features)
        scoring_done=time.monotonic()
        self.last_timings=dict(postprocess_ms=(scoring_done-native_done)*1000)
        if isinstance(self.scorer,NPUScorer):
            self.last_timings.pop('postprocess_ms')
            self.last_timings.update(self.scorer.last_timings)
        if self.lib.padim_get_timings(self.handle,self.timings,3)==0:
            self.last_timings.update(zip(('input_ms','network_ms','output_ms'),self.timings))
            self.last_timings['input_ms']+=(ready-started)*1000
        return result

    def close(self):
        if isinstance(getattr(self,'scorer',None),NPUScorer):self.scorer.close()
        if self.handle:
            self.lib.padim_close(self.handle);self.handle=None


class ONNXBackend:
    def __init__(self,config):
        import onnxruntime as ort
        self.scorer=PadimScorer(config['statistics'])
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(config['onnx_model'],sess_options=options,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs()
        if len(inputs)!=1 or inputs[0].shape!=[1,3,256,256] or inputs[0].type!='tensor(float)':
            raise ValueError('Expected normalized float32 NCHW [1,3,256,256] ONNX input')
        self.name=inputs[0].name
        mapping={tuple(output.shape[1:]):output.name for output in self.session.get_outputs() if output.shape[0]==1}
        if set(mapping)!=set(SHAPES) or len(self.session.get_outputs())!=3:
            raise ValueError('ONNX must expose layer1/layer2/layer3 feature tensors')
        self.outputs=[mapping[shape] for shape in SHAPES];self.last_timings={}

    def run(self,image):
        started=time.monotonic()
        values=self.session.run(self.outputs,{self.name:image})
        done=time.monotonic()
        result=self.scorer([value[0] for value in values])
        self.last_timings=dict(input_ms=0.,network_ms=(done-started)*1000,output_ms=0.,postprocess_ms=(time.monotonic()-done)*1000)
        return result

    def close(self): self.session=None
