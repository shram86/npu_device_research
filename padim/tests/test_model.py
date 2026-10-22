import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from padim_monitor.model import NBBackend,ONNXBackend,Preprocessor,SHAPES,MEAN,STD,resize_bilinear_chw,view_geometry
from padim_monitor.pipeline import load_config

ROOT=Path(__file__).resolve().parents[2]


class ReferenceTests(unittest.TestCase):
    def test_preprocessing_matches_experiment_b(self):
        from padim.padim_test_b.test_b import crop_image
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            for width,height in [(1280,720),(601,333),(333,601)]:
                rgb=np.random.default_rng(42).integers(0,256,(height,width,3),np.uint8)
                path=Path(directory)/'frame.png';Image.fromarray(rgb).save(path)
                expected=((crop_image(path)-MEAN)/STD).transpose(2,0,1)[None]
                actual=Preprocessor()(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                np.testing.assert_array_equal(actual,expected)
        roi=view_geometry(1280,720)
        self.assertLess(roi['width'],1280);self.assertEqual(roi['height'],720)

    def test_configurable_crop_position(self):
        from padim_monitor.model import resize_geometry
        from PIL import Image
        rng=np.random.default_rng(19)
        for width,height in [(1280,720),(333,601)]:
            frame=rng.integers(0,256,(height,width,3),np.uint8)
            for position in ({'x':0.,'y':0.},{'x':1.,'y':1.},{'x':.3,'y':.7}):
                nw,nh,left,top=resize_geometry(width,height,position)
                image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image=image.resize((nw,nh),Image.Resampling.BILINEAR).crop((left,top,left+256,top+256))
                expected=((np.asarray(image,dtype=np.float32)/np.float32(255)-MEAN)/STD).transpose(2,0,1)[None]
                np.testing.assert_array_equal(Preprocessor(position=position)(frame),expected)
                roi=view_geometry(width,height,position)
                self.assertEqual(roi['x'],round(left*width/nw))
                self.assertEqual(roi['y'],round(top*height/nh))
        for bad in ({'x':-1,'y':.5},{'x':1.1,'y':0},{'x':float('nan'),'y':0},{'x':True,'y':0},{'x':0}):
            with self.assertRaises(ValueError):Preprocessor(position=bad)

    def test_feature_resize_matches_torch(self):
        import torch
        import torch.nn.functional as F
        for size in (16,32):
            value=np.random.default_rng(size).random((7,size,size),dtype=np.float32)
            expected=F.interpolate(torch.from_numpy(value[None]),(64,64),mode='bilinear',align_corners=False)[0].numpy()
            np.testing.assert_allclose(resize_bilinear_chw(value),expected,atol=2e-7,rtol=1e-6)

    def test_real_onnx_and_scoring_match_experiment_b(self):
        from padim.padim_test_b.test_b import resize_bilinear_chw as reference_resize
        cfg=load_config(ROOT/'padim_monitor/config/config.json')
        backend=ONNXBackend(cfg)
        try:
            path=next(iter(sorted((ROOT/'padim/room_dataset/validation/good').glob('*.ppm'))))
            image=Preprocessor()(cv2.imread(str(path)))
            values=backend.session.run(backend.outputs,{backend.name:image})
            features=[value[0] for value in values]
            actual,score=backend.scorer(features)
            reference=np.concatenate([features[0],reference_resize(features[1]),reference_resize(features[2])])[backend.scorer.indices]
            delta=reference.transpose(1,2,0).reshape(4096,100)-backend.scorer.mean
            d2=np.einsum('pc,pcd,pd->p',delta,backend.scorer.precision,delta,optimize=True)
            expected=gaussian_filter(np.sqrt(np.maximum(d2,0)).reshape(64,64),sigma=2)
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-5)
            self.assertAlmostEqual(score,float(expected.max()),places=4)
        finally:backend.close()


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory(dir='/tmp');cls.root=Path(cls.storage.name)
        cls.library=cls.root/'libpadim_camera.so'
        fixtures=Path(__file__).parent
        subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                        '-I'+str(fixtures),str(ROOT/'padim_monitor/padim_camera.c'),str(fixtures/'mock_vip.c'),
                        '-lm','-o',str(cls.library)],check=True)
        meta=dict(Inputs={'input':dict(shape=[1,3,256,256],format='nchw',quantize=dict(qtype='i8',scale=.02,zero_point=-14))},Outputs={})
        for index,shape in enumerate(SHAPES):meta['Outputs'][str(index)]=dict(shape=[1,*shape],format='nchw',quantize=dict(qtype='i8',scale=1/128,zero_point=-128))
        cls.meta=cls.root/'metadata.json';cls.meta.write_text(json.dumps(meta))

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()

    def backend(self,name='model'):
        path=self.root/name;path.touch()
        cfg=load_config(ROOT/'padim_monitor/config/config.json')
        cfg.update(model=str(path),metadata=str(self.meta),library=str(self.library),scoring="cpu")
        class Scorer:
            def __init__(self,path):pass
            def bind_native(self,library):pass
            def __call__(self,features):
                for index,(feature,shape) in enumerate(zip(features,SHAPES)):
                    if feature.shape!=shape or feature.ravel()[0]!=(index+1)*.5:
                        raise AssertionError('Feature ordering/dequantization mismatch')
                return np.zeros((64,64),np.float32),0.
        with patch('padim_monitor.model.PadimScorer',Scorer):return NBBackend(cfg)

    def test_output_permutation_integer_and_floating_no_double_dequantization(self):
        for name in ('model','floating'):
            backend=self.backend(name)
            try:
                image=np.random.default_rng(5).random((1,3,256,256),dtype=np.float32)*4-2
                anomaly_map,score=backend.run(image)
                self.assertEqual(score,0.)
                expected=np.clip(np.rint(image/.02-14),-128,127).astype(np.int8)
                backend.lib.mock_input.argtypes=[ctypes.c_uint];backend.lib.mock_input.restype=ctypes.c_float
                for index in (0,100,65536,196607):self.assertEqual(backend.lib.mock_input(index),int(expected.ravel()[index]))
                self.assertIn('postprocess_ms',backend.last_timings)
            finally:backend.close()

    def test_selected_readback_matches_full_decode_then_resize(self):
        rng=np.random.default_rng(55)
        # Boundaries, all three layers, arbitrary saved order, no duplicates.
        indices=np.r_[0,63,64,191,192,447,rng.choice(np.setdiff1d(np.arange(448),[0,63,64,191,192,447]),94,replace=False)].astype(np.int32)
        fp=ctypes.POINTER(ctypes.c_float);ip=ctypes.POINTER(ctypes.c_int32)
        for name in ('model','floating'):
            backend=self.backend(name)
            try:
                lib=backend.lib
                lib.padim_set_channels.argtypes=[ctypes.c_void_p,ip,ctypes.c_size_t]
                lib.padim_set_channels.restype=ctypes.c_int
                lib.padim_run_selected.argtypes=lib.padim_run.argtypes
                lib.padim_run_selected.restype=ctypes.c_int
                image=np.zeros((1,3,256,256),np.float32)
                backend.run(image)
                full=np.concatenate([resize_bilinear_chw(value) for value in backend.features])[indices]
                output=np.empty((100,64,64),np.float32)
                def run():
                    return lib.padim_run_selected(backend.handle,backend.quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        backend.quantized.size,output.ctypes.data_as(fp),output.size,backend.error,len(backend.error))
                self.assertNotEqual(run(),0) # No selection configured.
                self.assertEqual(lib.padim_set_channels(backend.handle,indices.ctypes.data_as(ip),100),0)
                self.assertEqual(run(),0)
                np.testing.assert_allclose(output,full,rtol=2e-6,atol=2e-6)
                for bad in (np.full(100,-1,np.int32),np.full(100,448,np.int32),np.zeros(100,np.int32)):
                    self.assertNotEqual(lib.padim_set_channels(backend.handle,bad.ctypes.data_as(ip),100),0)
                self.assertEqual(run(),0) # Invalid setup does not corrupt last valid order.
                np.testing.assert_allclose(output,full,rtol=2e-6,atol=2e-6)
            finally:backend.close()

    def test_npu_score_graph_fp16_contract_and_lifecycle(self):
        from padim_monitor.model import NPUScorer
        cfg=load_config(ROOT/'padim_monitor/config/config.json')
        cfg.update(model=str(self.root/'feature'),metadata=str(self.meta),library=str(self.library),scoring='cpu')
        Path(cfg['model']).touch()
        backend=NBBackend(cfg)
        score=None
        try:
            features=[np.full(shape,(i+1)*.5,np.float32) for i,shape in enumerate(SHAPES)]
            # Choose one channel from each layer first, preserving saved order.
            indices=np.r_[0,64,192,np.arange(1,64),np.arange(65,99)]
            stats=self.root/'score_stats.npz'
            np.savez(stats,mean=np.zeros((4096,100),np.float32),indices=indices)
            cfg['statistics']=str(stats)
            for name in ('score','score_runfail','score_floating'):
                cfg['score_model']=str(self.root/name);Path(cfg['score_model']).touch()
                score=NPUScorer(cfg,backend.lib,backend.handle)
                self.assertIsNone(score.precision)
                if name=='score_runfail':
                    with self.assertRaisesRegex(RuntimeError,'VIP status'):score(features)
                elif name=='score_floating':
                    with self.assertRaisesRegex(ValueError,'Nonfinite'):score(features)
                else:
                    actual,value=score(features)
                    np.testing.assert_array_equal(actual,np.full((64,64),2,np.float32))
                    self.assertEqual(value,2.)
                    self.assertIn('score_network_ms',score.last_timings)
                score.close();score.close();score=None
                # Closing a second network must not destroy the feature runtime.
                backend.run(np.zeros((1,3,256,256),np.float32))
            cfg['score_model']=cfg['model']
            with self.assertRaisesRegex(RuntimeError,'Score NB must'):NPUScorer(cfg,backend.lib,backend.handle)
            backend.close()
            cfg.update(scoring='npu',score_model=str(self.root/'score'))
            backend=NBBackend(cfg)
            actual,value=backend.run(np.zeros((1,3,256,256),np.float32))
            self.assertEqual(value,2.)
            self.assertNotIn('postprocess_ms',backend.last_timings)
            self.assertIn('score_network_ms',backend.last_timings)
            self.assertIn('network_ms',backend.last_timings)
        finally:
            if score:score.close()
            backend.close()

    def test_native_distance_matches_real_statistics(self):
        from padim_monitor.model import PadimScorer
        scorer=PadimScorer(ROOT/'padim_monitor/models/padim_statistics.npz')
        rng=np.random.default_rng(12)
        features=[rng.standard_normal(shape,dtype=np.float32) for shape in SHAPES]
        expected,score=scorer(features)
        scorer.bind_native(ctypes.CDLL(str(self.library)))
        actual,native_score=scorer(features)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        self.assertAlmostEqual(native_score,score,delta=max(2e-5,abs(score)*2e-5))

    def test_native_preprocessing_center_crop_normalization_stride_and_reuse(self):
        native=Preprocessor('neon',self.library)
        rng=np.random.default_rng(9)
        for width,height,position in [(1280,720,{'x':.5,'y':.5}),(601,333,{'x':0.,'y':.5}),
                                      (601,333,{'x':1.,'y':.5}),(333,601,{'x':.5,'y':0.}),
                                      (333,601,{'x':.5,'y':1.}),(601,333,{'x':.3,'y':.7}),
                                      (1,1,{'x':.5,'y':.5})]:
            native=Preprocessor('neon',self.library,position)
            frame=rng.integers(0,256,(height,width+7,3),np.uint8)[:,:width]
            actual=native(frame)
            from padim_monitor.model import resize_geometry
            nw,nh,left,top=resize_geometry(width,height,position)
            xx=np.clip((np.arange(256,dtype=np.float32)+np.float32(left+.5))*np.float32(width/nw)-np.float32(.5),0,width-1)
            yy=np.clip((np.arange(256,dtype=np.float32)+np.float32(top+.5))*np.float32(height/nh)-np.float32(.5),0,height-1)
            x0,y0=np.floor(xx).astype(int),np.floor(yy).astype(int)
            x1,y1=np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
            wx=(xx-x0).astype(np.float32)[None,:,None];wy=(yy-y0).astype(np.float32)[:,None,None]
            rgb=frame[:,:,::-1].astype(np.float32)
            a=rgb[y0[:,None],x0[None,:]];b=rgb[y0[:,None],x1[None,:]]
            c=rgb[y1[:,None],x0[None,:]];d=rgb[y1[:,None],x1[None,:]]
            upper=a+wx*(b-a);lower=c+wx*(d-c)
            expected=(((upper+wy*(lower-upper))*np.float32(1/255)-MEAN)/STD).transpose(2,0,1)[None]
            # Shifted fractional coordinates can differ by float32/FMA rounding.
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=2e-6)
            self.assertIs(actual,native.output)
            self.assertTrue(actual.flags.c_contiguous)
        self.assertIs(native(frame),actual)
        with self.assertRaisesRegex(ValueError,'stride'):native(frame[::-1])
        with self.assertRaisesRegex(RuntimeError,'rebuild'):Preprocessor('neon',object())

    def test_runtime_failures(self):
        backend=self.backend('runfail')
        try:
            with self.assertRaisesRegex(RuntimeError,'-99'):backend.run(np.zeros((1,3,256,256),np.float32))
        finally:backend.close()

    def test_matching_metadata_is_required(self):
        original=self.meta.read_text();wrong=json.loads(original);wrong['Inputs']['input']['quantize']['scale']=.03
        self.meta.write_text(json.dumps(wrong))
        try:
            with self.assertRaisesRegex(ValueError,'quantization mismatch'):self.backend()
        finally:self.meta.write_text(original)


if __name__=='__main__':unittest.main()
