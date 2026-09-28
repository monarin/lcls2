"""Read current BEAM configuration and preserve its exact contents locally."""
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from psdaq.configdb.configdb import configdb
from psdaq.configdb.get_config import get_config_with_params
url='https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws'
db=configdb(url,'ued',False,'configDB')
config=get_config_with_params(url,'ued','configDB','BEAM','epixquad1kfps_0')
again=get_config_with_params(url,'ued','configDB','BEAM','epixquad1kfps_0')
if config != again:
    raise RuntimeError('Configuration changed between two reads')
folder=Path(__file__).parent
raw=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
(folder/'beam-config.json.gz').write_bytes(gzip.compress(raw,mtime=0))
metadata={'utc':datetime.now(timezone.utc).isoformat(),'url':url,'hutch':'ued',
          'alias':'BEAM','device':'epixquad1kfps_0','key':None,
          'key_note':'get_key returned HTTP 403; two configuration reads matched; content identified by SHA256',
          'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
          'config_reference':config.get('_cfgTypeRef'),
          'use_serial_db':config.get('use_serial_db',False),
          'expert_blocks':list(config['expert']['EpixQuad']),
          'camera_summary':{k:v for k,v in config['expert']['EpixQuad'].items()
                            if not k.startswith('Epix10kaSaci')},
          'pcie':config['expert'].get('DevPcie'),
          'user':{k:v for k,v in config['user'].items() if k!='pixel_map_raw'}}
(folder/'config-summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
