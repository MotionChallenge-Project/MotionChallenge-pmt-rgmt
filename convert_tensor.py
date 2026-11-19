"""Extract the released RGMT actor and validate dynamic batch inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from network import RGMTActor
from motion_challenge.validation import sha256_file


def convert(source, output, expected, device="cuda:0"):
    source, output = Path(source), Path(output)
    if sha256_file(source) != expected: raise ValueError("Checkpoint hash mismatch")
    checkpoint = torch.load(source,map_location="cpu",weights_only=True)
    signature = checkpoint["policy_metadata"]["signature"]
    for key,value in {"actor_obs_dim":93,"history_len":10,"cmd_len":21,"use_vel_estimator":False,"use_anchor_estimator":False}.items():
        if signature[key] != value: raise ValueError("Unexpected signature: "+key)
    original = RGMTActor(checkpoint["model_state_dict"]).eval()
    output.mkdir(parents=True,exist_ok=True)
    save_file({k:v.contiguous() for k,v in original.state_dict().items()},str(output/"actor.safetensors"))
    metadata = {"source_sha256":expected,"weights_sha256":sha256_file(output/"actor.safetensors"),"signature":signature,
                "format":"safetensors","network":"network:RGMTActor","network_sha256":sha256_file(Path(__file__).with_name("network.py"))}
    (output/"conversion.json").write_text(json.dumps(metadata,indent=2)+"\n")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1739)
    rows=[];maximum=0.
    for batch in (1,7,40,256):
        inputs=[torch.randn(batch,*shape) for shape in ((93,),(10,93),(21,38))]
        with torch.inference_mode():expected_output=torch.cat([original(*(x[i:i+1] for x in inputs)) for i in range(batch)])
        for device in ("cpu",device):
            actor=RGMTActor(load_file(str(output/"actor.safetensors"))).to(device).eval()
            with torch.inference_mode():actual=actor(*(x.to(device) for x in inputs)).cpu()
            np.testing.assert_allclose(actual.numpy(),expected_output.numpy(),rtol=2e-5,atol=2e-5)
            error=float((actual-expected_output).abs().max());maximum=max(maximum,error)
            rows.append({"batch":batch,"device":device,"max_abs_error":error})
    report={"status":"passed","source_sha256":metadata["source_sha256"],"weights_sha256":metadata["weights_sha256"],
            "reference":"released checkpoint actor evaluated independently per sample on CPU","cases":rows,"max_abs_error":maximum}
    (output/"parity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path);parser.add_argument("--sha256",required=True);parser.add_argument("--device",default="cuda:0")
    a=parser.parse_args();convert(a.source,a.output,a.sha256,a.device)
