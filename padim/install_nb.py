#!/usr/bin/env python3
"""Install an explicitly selected NB/metadata pair and record its provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .model import read_metadata
else:
    from model import read_metadata


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path(__file__).with_name('models'))
    args=parser.parse_args()
    if not args.model.is_file():parser.error('NB file not found')
    read_metadata(args.metadata)
    targets=[args.destination/name for name in ('network_binary.nb','nbg_meta.json','nb_manifest.json')]
    if any(path.exists() for path in targets):parser.error('Destination NB/metadata/manifest already exists; choose a fresh destination')
    args.destination.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_model=str(args.model.resolve()),source_metadata=str(args.metadata.resolve()),
                  model_sha256=sha256(args.model),metadata_sha256=sha256(args.metadata),
                  historical_experiment_build_verified=False,
                  note='Explicit selection only: the build used for the historical AUROC 0.990625 was not recorded.')
    created=[]
    try:
        for source,target in zip((args.model,args.metadata),targets):
            with source.open('rb') as src,target.open('xb') as dst:
                created.append(target);shutil.copyfileobj(src,dst)
        with targets[2].open('x') as stream:
            created.append(targets[2]);json.dump(manifest,stream,indent=2);stream.write('\n')
    except Exception:
        for path in created:path.unlink(missing_ok=True)
        raise
    print(f'Installed NB/metadata into {args.destination}. Recheck score quality and calibrate a device threshold.')


if __name__=='__main__':main()
