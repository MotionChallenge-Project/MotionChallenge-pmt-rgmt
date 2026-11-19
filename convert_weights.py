"""One-command safe-weight preparation."""
import json
from pathlib import Path
from convert_tensor import convert
from motion_challenge.validation import sha256_file


def prepare(root,runtime,source,output,artifacts,device):
    output=Path(output)
    receipt=output/"conversion.json"
    if receipt.exists() and (output/"parity.json").exists():
        meta=json.loads(receipt.read_text())
        if (meta.get("source_sha256")==runtime["weights_sha256"]
            and meta.get("network_sha256")==sha256_file(Path(root)/"network.py")
            and sha256_file(output/"actor.safetensors")==meta["weights_sha256"]):
            return output
    convert(source,output,runtime["weights_sha256"],device)
    return output
