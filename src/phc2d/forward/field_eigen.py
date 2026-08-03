"""Data-only supervised TE Bloch-eigenfield training and evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv, hashlib, json, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.models import architecture_fingerprint, build_model, describe_model_identity, get_model_spec
from phc2d.physics import canonicalize_complex_efield, evaluate_te_bloch_eigen_residual
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.training import DeviceConfig, collect_device_metadata, peak_gpu_memory_mb, resolve_device, synchronize_device


ROOT = Path(__file__).resolve().parents[3]


def is_field_eigen_config(path: str | Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return str(dict(raw.get("experiment") or {}).get("execution", "")).lower() == "field_eigen_supervised"


def _torch():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    return torch, DataLoader, TensorDataset


def _hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    import h5py
    with h5py.File(path, "r") as h:
        splits=np.asarray(h["source_split"]).astype(str); ids=np.asarray(h["record_id"]).astype(str); source_ids=np.asarray(h["source_sample_id"]).astype(str)
        e=np.asarray(h["efield_complex"]); hh=np.asarray(h["hfield_complex"]); eps=np.asarray(h["epsilon"],dtype=np.float32)
        k=np.asarray(h["k_points"],dtype=np.float32); band=np.asarray(h["band_index"],dtype=np.float32); freq=np.asarray(h["eigenfrequency"],dtype=np.float32)
        dx=np.asarray(h["dx"],dtype=np.float64); dy=np.asarray(h["dy"],dtype=np.float64)
    if e.shape[:3] != hh.shape[:3] or e.shape[0] != eps.shape[0]: raise ValueError("Field HDF5 E/H/epsilon sample axes are inconsistent.")
    nx,ny=int(e.shape[1]),int(e.shape[2])
    if nx % eps.shape[1] or ny % eps.shape[2]: raise ValueError("epsilon-to-field grid ratio is not integral.")
    channel=lambda z: np.concatenate((z.real.squeeze(3),z.imag.squeeze(3)),axis=-1).transpose(0,3,1,2).astype(np.float32)
    target=np.concatenate((channel(e),channel(hh)),axis=1)
    e_min,e_max=(float(v) for v in dict(cfg.get("normalization") or {}).get("epsilon_range",[1.0,8.9]))
    eps_grid=np.repeat(np.repeat((eps-e_min)/(e_max-e_min),nx//eps.shape[1],axis=1),ny//eps.shape[2],axis=2)
    k_scale=float(dict(cfg.get("normalization") or {}).get("k_scale",0.5)); br=dict(cfg.get("normalization") or {}).get("band_range_one_based",[2,10])
    context=np.stack((k[:,0]/k_scale,k[:,1]/k_scale,(band-float(br[0]))/max(float(br[1])-float(br[0]),1.0)),axis=1)
    x=np.concatenate((eps_grid[:,None],np.broadcast_to(context[:,:,None,None],(len(eps),3,nx,ny))),axis=1).astype(np.float32)
    idx={name:np.flatnonzero(splits==name).astype(np.int64) for name in ("train","val","test")}
    if {k:len(v) for k,v in idx.items()} != {"train":750,"val":125,"test":125}: raise ValueError("Field dataset must use the frozen 750/125/125 source_split.")
    source_sets={name:set(source_ids[values].tolist()) for name,values in idx.items()}
    if source_sets["train"] & source_sets["val"] or source_sets["train"] & source_sets["test"] or source_sets["val"] & source_sets["test"]: raise ValueError("source_sample_id leakage across field source_split.")
    return {"x":x,"target":target,"freq":freq,"epsilon":eps,"k":k,"dx":dx,"dy":dy,"ids":ids,"idx":idx,"field_shape":e.shape[1:],"dataset_hash":_hash(path)}


def _loss(torch: Any, field: Any, freq: Any, y: Any, f: Any) -> Any:
    e=((field[:,:6]-y[:,:6]).square().flatten(1).sum(1)/y[:,:6].square().flatten(1).sum(1).clamp_min(1e-12)).mean()
    h=((field[:,6:]-y[:,6:]).square().flatten(1).sum(1)/y[:,6:].square().flatten(1).sum(1).clamp_min(1e-12)).mean()
    return 0.5*e+0.5*h+0.1*(freq-f).square().mean()


def _metrics(data: dict[str,Any], field: np.ndarray, freq: np.ndarray, indices: np.ndarray) -> tuple[dict[str,float],list[dict[str,Any]]]:
    y=data["target"][indices]; eps=data["epsilon"][indices]; k=data["k"][indices]; dx=data["dx"][indices]; dy=data["dy"][indices]
    rel=lambda a,b: np.sqrt(np.sum((a-b)**2,axis=(1,2,3)))/np.maximum(np.sqrt(np.sum(a**2,axis=(1,2,3))),1e-12)
    er,hr=rel(y[:,:6],field[:,:6]),rel(y[:,6:],field[:,6:])
    def complex_part(z,offset): return z[:,offset:offset+3].transpose(0,2,3,1)+1j*z[:,offset+3:offset+6].transpose(0,2,3,1)
    et,ep=complex_part(y,0),complex_part(field,0); amp=np.mean(np.abs(np.abs(ep)-np.abs(et)),axis=(1,2,3)); mask=np.abs(et)>=0.05*np.max(np.abs(et),axis=(1,2,3),keepdims=True); phase=np.angle(np.exp(1j*(np.angle(ep)-np.angle(et)))); phase_mae=np.array([np.mean(np.abs(v[m])) if np.any(m) else np.nan for v,m in zip(phase,mask)])
    residual=[]; boundary=[]; invalid=0; rows=[]
    for pos,original in enumerate(indices):
        efield=ep[pos][:,:,None,:]
        try:
            canon=canonicalize_complex_efield(efield,epsilon=eps[pos],dx=float(dx[pos]),dy=float(dy[pos])); r=evaluate_te_bloch_eigen_residual(canon,epsilon=eps[pos],eigenfrequency=float(freq[pos]),dx=float(dx[pos]),dy=float(dy[pos]),bloch_k_point=k[pos]); residual.append(r["relative_residual"]); boundary.append(r["boundary_relative_residual"])
        except ValueError: invalid+=1; residual.append(float("nan")); boundary.append(float("nan"))
        rows.append({"record_id":str(data["ids"][original]),"e_relative_l2":float(er[pos]),"h_relative_l2":float(hr[pos]),"amplitude_mae":float(amp[pos]),"phase_mae_rad":float(phase_mae[pos]),"frequency_mre_pct":float(100*abs(freq[pos]-data["freq"][original])/max(abs(data["freq"][original]),1e-12)),"maxwell_relative_residual":float(residual[-1]),"boundary_relative_residual":float(boundary[-1])})
    return {"e_relative_l2":float(np.mean(er)),"h_relative_l2":float(np.mean(hr)),"amplitude_mae":float(np.mean(amp)),"phase_mae_rad":float(np.nanmean(phase_mae)),"frequency_mre_pct":float(np.mean([r["frequency_mre_pct"] for r in rows])),"maxwell_relative_residual":float(np.nanmean(residual)),"boundary_relative_residual":float(np.nanmean(boundary)),"invalid_predicted_field_count":float(invalid)},rows


def _read(path: str|Path) -> dict[str,Any]:
    p=Path(path); raw=yaml.safe_load(p.read_text(encoding="utf-8")) or {}; raw["_path"]=p; return raw


def _run(raw: dict[str,Any], *, run_dir: Path|None=None, checkpoint: Path|None=None, train: bool=True) -> Path:
    torch,DataLoader,TensorDataset=_torch(); cfg=raw; dataset=ROOT/str(dict(cfg["dataset"])["path"]); model_cfg=dict(cfg["model"]); train_cfg=dict(cfg["training"]); runtime_cfg=dict(cfg["runtime"]); seed=int(train_cfg.get("seed",0)); torch.manual_seed(seed); np.random.seed(seed)
    device_info=resolve_device(DeviceConfig.from_mapping(runtime_cfg),torch_module=torch)
    data=_load(dataset,cfg); freq_mean=float(data["freq"][data["idx"]["train"]].mean()); freq_std=max(float(data["freq"][data["idx"]["train"]].std()),1e-8); device=device_info.torch_device
    exp=str(dict(cfg["experiment"])["experiment_id"]); out=make_run_dir(exp,runs_dir=ROOT/str(dict(cfg.get("output") or {}).get("runs_dir","runs")),run_dir=run_dir); logger=setup_logging(log_file=out/"run.log",run_meta={"stage":"field_eigen_supervised"}); ensure_figures_dir(out); started=time.perf_counter()
    model=build_model(str(model_cfg["name"]),**dict(model_cfg.get("params") or {})).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(train_cfg.get("learning_rate",3e-4)),weight_decay=float(train_cfg.get("weight_decay",1e-5)))
    if checkpoint:
        payload=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(payload["state_dict"]); freq_mean=float(payload["freq_mean"]); freq_std=float(payload["freq_std"])
    history=[]; best=float("inf"); best_path=out/"best_val.pt"; x=torch.as_tensor(data["x"]); y=torch.as_tensor(data["target"]); f=torch.as_tensor((data["freq"]-freq_mean)/freq_std)
    if train:
        loader=DataLoader(TensorDataset(x[data["idx"]["train"]],y[data["idx"]["train"]],f[data["idx"]["train"]]),batch_size=int(train_cfg.get("batch_size",64)),shuffle=True,pin_memory=device_info.pin_memory)
        for epoch in range(1,int(train_cfg.get("epochs",200))+1):
            model.train(); losses=[]
            for xb,yb,fb in loader:
                xb,yb,fb=xb.to(device),yb.to(device),fb.to(device); opt.zero_grad(set_to_none=True); pf,pfr=model(xb); loss=_loss(torch,pf,pfr,yb,fb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            model.eval();
            with torch.no_grad(): vf,vfr=model(x[data["idx"]["val"]].to(device)); val=float(_loss(torch,vf,vfr,y[data["idx"]["val"]].to(device),f[data["idx"]["val"]].to(device)).cpu())
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":val})
            if val<best: best=val; torch.save({"state_dict":model.state_dict(),"model_name":model_cfg["name"],"model_params":dict(model_cfg.get("params") or {}),"freq_mean":freq_mean,"freq_std":freq_std,"dataset_hash":data["dataset_hash"],"seed":seed},best_path)
        model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["state_dict"])
    model.eval(); test=data["idx"][str(dict(cfg.get("evaluation") or {}).get("split","test"))]
    sync=time.perf_counter();
    with torch.no_grad(): pf,pfr=model(x[test].to(device)); synchronize_device(device,torch_module=torch)
    eval_s=time.perf_counter()-sync; pred=pf.cpu().numpy(); freq=pfr.cpu().numpy()*freq_std+freq_mean; metrics,rows=_metrics(data,pred,freq,test); metrics.update({"train_time_s":float(time.perf_counter()-started-eval_s) if train else 0.0,"eval_time_s":eval_s,"inference_time_per_sample_ms":1000*eval_s/max(len(test),1),"wall_time_total_s":time.perf_counter()-started,"parameter_count":sum(p.numel() for p in model.parameters()),"peak_gpu_memory_mb":peak_gpu_memory_mb(device,torch_module=torch),"n_train":750,"n_val":125,"n_test":125,"seed":seed,"lambda_pde":0.0})
    torch.save({"state_dict":model.state_dict(),"model_name":model_cfg["name"],"model_params":dict(model_cfg.get("params") or {}),"freq_mean":freq_mean,"freq_std":freq_std,"dataset_hash":data["dataset_hash"],"seed":seed},out/"model.pt")
    with (out/"history.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=["epoch","train_loss","val_loss"]);w.writeheader();w.writerows(history)
    with (out/"per_case_metrics.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_resolved_config(out,yaml.safe_dump({k:v for k,v in cfg.items() if k!="_path"},sort_keys=False,allow_unicode=True));write_metrics(out,metrics); write_json(out/"model_identity.json",describe_model_identity(model,str(model_cfg["name"]),spec=get_model_spec(str(model_cfg["name"])))); write_results(out,{"status":"supervised_field_baseline_pilot","stage":"field_eigen_train" if train else "field_eigen_eval","model_name":model_cfg["name"],"checkpoint_path":str(out/"model.pt"),"dataset_hash":data["dataset_hash"],"architecture_fingerprint":architecture_fingerprint(model),"lambda_pde":0.0,"metrics":metrics}); write_run_meta(out,{**collect_device_metadata(device_info),"status":"supervised_field_baseline_pilot"}); logger.info("Completed field-eigen run %s",out); return out


def run_field_eigen_train(path: str|Path) -> Path: return _run(_read(path),train=True)
def run_field_eigen_eval(path: str|Path) -> Path:
    raw=_read(path); ck=Path(str(dict(raw["model"])["checkpoint_path"])); return _run(raw,checkpoint=ck,train=False)

def run_field_eigen_benchmark(path: str|Path) -> Path:
    raw=_read(path); parent=make_run_dir(str(dict(raw["experiment"])["experiment_id"]),runs_dir=ROOT/str(dict(raw.get("output") or {}).get("runs_dir","runs"))); rows=[]
    logger=setup_logging(log_file=parent/"run.log",run_meta={"stage":"field_eigen_supervised_benchmark"});ensure_figures_dir(parent)
    for model in list(dict(raw["benchmark"])["models"]):
        for seed in list(dict(raw["training"])["seeds"]):
            case={**raw,"model":{"name":model,"params":dict(dict(raw["benchmark"]).get("model_params",{})).get(model,{})},"training":{**dict(raw["training"]),"seed":int(seed)}}; run=_run(case,run_dir=parent/"cases"/f"{model}_seed{seed}"); metrics=json.loads((run/"metrics.json").read_text()); rows.append({"model":model,"seed":seed,"run_dir":str(run),**{k:metrics[k] for k in ("e_relative_l2","h_relative_l2","amplitude_mae","phase_mae_rad","frequency_mre_pct","maxwell_relative_residual","boundary_relative_residual","parameter_count")}});logger.info("Completed field-eigen case model=%s seed=%s e_relative_l2=%.6f",model,seed,float(metrics["e_relative_l2"]))
    with (parent/"leaderboard.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (parent/"history.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={m:{k:{"mean":float(np.mean([r[k] for r in rows if r["model"]==m])),"std":float(np.std([r[k] for r in rows if r["model"]==m]))} for k in ("e_relative_l2","h_relative_l2","amplitude_mae","phase_mae_rad","frequency_mre_pct","maxwell_relative_residual","boundary_relative_residual")} for m in sorted({r["model"] for r in rows})}; write_resolved_config(parent,yaml.safe_dump({k:v for k,v in raw.items() if k!="_path"},sort_keys=False));write_metrics(parent,{"models":summary,"case_count":len(rows)});write_results(parent,{"status":"supervised_field_baseline_pilot","case_count":len(rows),"leaderboard":"leaderboard.csv","models":summary});write_run_meta(parent,{"status":"supervised_field_baseline_pilot"});return parent
