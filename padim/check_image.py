#!/usr/bin/env python3
"""Run the monitor's exact image preprocessing, feature extraction and PaDiM scoring."""
import argparse
import json
from pathlib import Path

if __package__:
    from .pipeline import load_config
    from .model import NBBackend,ONNXBackend,Preprocessor
else:
    from pipeline import load_config
    from model import NBBackend,ONNXBackend,Preprocessor
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',type=Path,nargs='+')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config')/'config.json')
    parser.add_argument('--compute',choices=['npu','cpu'],default='npu')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--preprocessing',choices=['pil','neon'])
    parser.add_argument('--library',type=Path)
    parser.add_argument('--scoring',choices=['cpu','npu'])
    parser.add_argument('--score-model',type=Path)
    parser.add_argument('--score-metadata',type=Path)
    parser.add_argument('--output-dir',type=Path,help='Optional per-image .npy map output directory')
    args=parser.parse_args();cfg=load_config(args.config)
    cfg['backend']='nb' if args.compute=='npu' else 'onnx'
    if args.model:cfg['model' if cfg['backend']=='nb' else 'onnx_model']=str(args.model.resolve())
    if args.metadata:cfg['metadata']=str(args.metadata.resolve())
    if args.library:cfg['library']=str(args.library.resolve())
    if args.preprocessing:cfg['preprocessing']=args.preprocessing
    if args.scoring:cfg['scoring']=args.scoring
    if cfg['backend']=='onnx':cfg['scoring']='cpu'
    for name in ('score_model','score_metadata'):
        if getattr(args,name):cfg[name]=str(getattr(args,name).resolve())
    backend=(NBBackend if cfg['backend']=='nb' else ONNXBackend)(cfg)
    report=[]
    try:
        preprocessor=Preprocessor(cfg['preprocessing'],cfg['library'] if cfg['preprocessing']=='neon' else None,cfg.get('crop_position'))
        if args.output_dir:args.output_dir.mkdir(parents=True,exist_ok=True)
        for index,path in enumerate(args.images):
            frame=cv2.imread(str(path))
            if frame is None:raise ValueError(f'Cannot read {path}')
            anomaly_map,score=backend.run(preprocessor(frame))
            report.append(dict(image=str(path.resolve()),backend=cfg['backend'],scoring=cfg['scoring'],preprocessing=cfg['preprocessing'],score=score,
                               threshold=cfg['threshold'],threshold_calibrated=False,
                               above_threshold=score>=cfg['threshold'],timings_ms=backend.last_timings))
            if args.output_dir:np.save(args.output_dir/f'{index:05d}_{path.stem}.npy',anomaly_map)
    finally:backend.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
