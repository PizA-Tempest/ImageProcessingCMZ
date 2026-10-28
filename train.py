import argparse, json, random, copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

def load_bbox_map(path):
    if not Path(path).exists(): return {}
    bb = pd.read_csv(path).dropna(subset=['bbox']); m = {}
    for _, r in bb.iterrows():
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in str(r['bbox']).split(',')]
            m[r['image_path']] = (x1, y1, x2, y2)
        except Exception:
            pass
    return m

def crop_padded(im, box, pad=0.15):
    w, h = im.size; x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
    l, t, r, b = max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)
    if r - l < 8 or b - t < 8: return im
    return im.crop((l, t, r, b))

class FrameDataset(Dataset):
    def __init__(self, df, class_to_idx, transform, bbox_map=None):
        self.df=df.reset_index(drop=True); self.map=class_to_idx; self.tf=transform; self.bbox_map=bbox_map or {}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im: image=im.convert('RGB')
        bb=self.bbox_map.get(r.image_path)
        if bb: image=crop_padded(image, bb)
        return self.tf(image), self.map[r.label]

def dev_score(parts, classes, overall):
    """(class coverage across partitions, -distribution error, -row imbalance)."""
    cov=sum(len(set(p.label)&set(classes)) for p in parts)
    err=sum(sum(abs(p.label.value_counts(normalize=True).reindex(overall.index,fill_value=0)-overall)) for p in parts)
    sizes=[len(p) for p in parts]
    return (cov, -err, -abs(max(sizes)-min(sizes)))

def assign_folds(df, classes, n_folds, seed):
    """Assign whole videos to folds: balance row counts, maximize class coverage."""
    videos=df.source_video.unique().tolist()
    overall=df.label.value_counts(normalize=True)
    rows={v:int((df.source_video==v).sum()) for v in videos}
    best=None
    for attempt in range(2000):
        rng=np.random.RandomState(seed+attempt)
        order=sorted(videos, key=lambda v: -rows[v])
        # randomize ties
        order=[order[i] for i in rng.permutation(len(order))]
        order=sorted(order, key=lambda v: -rows[v])
        folds=[[] for _ in range(n_folds)]; loads=[0]*n_folds
        for v in order:
            mn=min(loads); cands=[k for k in range(n_folds) if loads[k]==mn]
            k=int(rng.choice(cands)); folds[k].append(v); loads[k]+=rows[v]
        parts=[df[df.source_video.isin(f)] for f in folds]
        key=dev_score(parts, classes, overall)
        if best is None or key>best[0]: best=(key, folds)
    return best[1]

def pick_val_videos(train_pool, classes, n_val_videos, seed, test_part):
    """Choose val videos from the pool maximizing coverage and class-mix balance."""
    overall=train_pool.label.value_counts(normalize=True)
    vids=train_pool.source_video.unique().tolist()
    n_val=min(max(1, n_val_videos), len(vids)-2) if len(vids)>2 else max(1, len(vids)-1)
    best=None
    for attempt in range(600):
        rng=np.random.RandomState(seed+attempt)
        perm=rng.permutation(len(vids))
        val_v={vids[i] for i in perm[:n_val]}
        val=train_pool[train_pool.source_video.isin(val_v)]
        train=train_pool[~train_pool.source_video.isin(val_v)]
        if len(val)<8 or len(train)<30: continue
        key=dev_score([train,val], classes, overall)
        if best is None or key>best[0]: best=(key, train, val)
    if best is None:
        # fall back: first videos as val
        val=train_pool[train_pool.source_video.isin(vids[:n_val])]
        train=train_pool[~train_pool.source_video.isin(vids[:n_val])]
        best=((0,0,0), train, val)
    return best[1], best[2]

def build_model(n_classes, device):
    try:
        model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception as e:
        print('Could not load pretrained weights:',e); model=models.efficientnet_b0(weights=None)
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,n_classes)
    for param in model.features.parameters(): param.requires_grad=False
    return model.to(device)

def train_fold(train, val, classes, args, bbox_map, device, tag=''):
    class_to_idx={c:i for i,c in enumerate(classes)}
    tf_train=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),transforms.RandomAffine(0,translate=(0.08,0.08)),
        transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.1),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tf_eval=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_loader=DataLoader(FrameDataset(train,class_to_idx,tf_train,bbox_map),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(FrameDataset(val,class_to_idx,tf_eval,bbox_map),batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=build_model(len(classes), device)
    cnt=train.label.value_counts()
    if args.weight_mode=='none':
        lw=pd.Series(1.0, index=classes)
    elif args.weight_mode=='sqrt':
        lw=((cnt.sum()/(len(classes)*cnt))**0.5).reindex(classes).fillna(1.0).clip(upper=3.0)
    else:
        lw=(cnt.sum()/(len(classes)*cnt)).reindex(classes).fillna(1.0).clip(upper=8.0)
    print(f'  {tag} class weights ({args.weight_mode}):', {c: round(float(lw[c]),2) for c in classes})
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(lw.values,dtype=torch.float32).to(device),label_smoothing=0.1)
    optimizer=torch.optim.AdamW(filter(lambda x:x.requires_grad,model.parameters()),lr=1e-3,weight_decay=0.01)
    unfreeze_after=3; unfrozen=False; best=-1.0; best_loss=float('inf'); best_state=None; history=[]
    for epoch in range(args.epochs):
        if epoch==unfreeze_after and not unfrozen:
            for param in model.features.parameters(): param.requires_grad=True
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
            unfrozen=True
        model.train(); train_loss=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step(); train_loss+=loss.item()*len(y)
        model.eval(); val_loss=0; vy=[]; vyp=[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); logits=model(x); val_loss+=criterion(logits,y).item()*len(y)
                vy.extend(y.cpu().numpy().tolist()); vyp.extend(logits.argmax(1).cpu().numpy().tolist())
        train_loss/=max(1,len(train_loader.dataset)); val_loss/=max(1,len(val_loader.dataset))
        val_f1=f1_score(vy,vyp,labels=list(range(len(classes))),average='macro',zero_division=0)
        history.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':val_loss,'val_macro_f1':val_f1})
        if val_f1>best or (val_f1==best and val_loss<best_loss):
            best=val_f1; best_loss=val_loss
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if args.verbose or epoch in (0, args.epochs-1):
            print(f'  {tag}epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}')
    if best_state: model.load_state_dict(best_state)
    return model, history, best

def predict_fold(model, test, classes, bbox_map, args):
    class_to_idx={c:i for i,c in enumerate(classes)}
    tf_eval=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    loader=DataLoader(FrameDataset(test,class_to_idx,tf_eval,bbox_map),batch_size=args.batch_size,shuffle=False,num_workers=0)
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loader:
            preds=model(x.to(device_global)).argmax(1)
            y_true.extend(y.numpy().tolist()); y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred

device_global=None

def main():
    global device_global
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',default='data/annotations.csv')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cv-folds',type=int,default=4,help='group k-fold over videos; 1 = single holdout')
    p.add_argument('--model-out',default='models/behavior_classifier.pt')
    p.add_argument('--results-dir',default='results')
    p.add_argument('--weight-mode',choices=['balanced','sqrt','none'],default='sqrt',help='class weight scheme')
    p.add_argument('--verbose',action='store_true')
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); device_global=device
    print('Device:',device)
    df=pd.read_csv(args.annotations).fillna('')
    required={'image_path','label','source_video'}
    if not required.issubset(df.columns): raise SystemExit(f'CSV must contain {sorted(required)}')
    EXCLUDE={'Unclear','Not Visible','No Civet',''}
    df=df[~df.label.astype(str).str.strip().isin(EXCLUDE)].copy()
    df=df[df.image_path.map(lambda x: Path(x).exists())]
    if len(df)<20: raise SystemExit('Need at least 20 labeled, readable images.')
    counts=df.label.value_counts(); classes=sorted(counts[counts>=3].index.tolist()); df=df[df.label.isin(classes)].reset_index(drop=True)
    if len(classes)<2: raise SystemExit('Need at least 2 behavior classes with 3+ images each.')
    bbox_map=load_bbox_map(Path(args.annotations).parent/'bboxes.csv')
    n_crop=int(df.image_path.isin(bbox_map).sum())
    print(f'Classes ({len(classes)}): {classes}')
    print(f'Labeled frames: {len(df)} | cropping to YOLO box: {n_crop}/{len(df)}')

    videos=df.source_video.unique().tolist()
    n_folds=max(1, min(args.cv_folds, len(videos)))
    if n_folds<2 or len(videos)<3:
        # legacy single holdout
        overall=df.label.value_counts(normalize=True)
        best_split=None; best_key=None
        for attempt in range(4000):
            rng=np.random.RandomState(args.seed+attempt)
            perm=rng.permutation(len(videos))
            n_test=max(1,round(len(videos)*0.2)); n_val=max(1,round(len(videos)*0.2))
            test_v={videos[i] for i in perm[:n_test]}; val_v={videos[i] for i in perm[n_test:n_test+n_val]}
            test=df[df.source_video.isin(test_v)].copy(); val=df[df.source_video.isin(val_v)].copy()
            train=df[~df.source_video.isin(test_v|val_v)].copy()
            if min(len(train),len(val),len(test))<8: continue
            key=dev_score([train,val,test],classes,overall)
            if best_key is None or key>best_key: best_key=key; best_split=(train,val,test)
        if best_split is None: raise SystemExit('Could not build a valid train/val/test split.')
        splits=[('holdout',)+best_split]
        print(f'Split: single group holdout, coverage={best_key[0]}/{3*len(classes)}')
    else:
        folds=assign_folds(df, classes, n_folds, args.seed)
        splits=[]
        for k in range(n_folds):
            test=df[df.source_video.isin(folds[k])].copy()
            pool=df[~df.source_video.isin(folds[k])]
            train,val=pick_val_videos(pool, classes, max(1,round(len(videos)*0.25)), args.seed+100*k, test)
            splits.append((f'fold{k+1}', train, val, test))
        print(f'Split: {n_folds}-fold group CV over {len(videos)} videos (each video is test exactly once)')

    fold_results=[]; pooled_true=[]; pooled_pred=[]
    best_val=-1.0; best_model_state=None
    for tag, train, val, test in splits:
        print(f'{tag}: train={len(train)} val={len(val)} test={len(test)} | test videos={sorted(test.source_video.unique())}')
        model, history, val_f1 = train_fold(train, val, classes, args, bbox_map, device, tag=tag)
        y_true, y_pred = predict_fold(model, test, classes, bbox_map, args)
        acc=accuracy_score(y_true,y_pred); macro=f1_score(y_true,y_pred,labels=list(range(len(classes))),average='macro',zero_division=0)
        fold_results.append({'fold':tag,'n_train':len(train),'n_val':len(val),'n_test':len(test),
            'test_videos':sorted(test.source_video.unique()),'accuracy':acc,'macro_f1':macro,
            'val_macro_f1':val_f1,'train_label_counts':train.label.value_counts().to_dict(),
            'test_label_counts':test.label.value_counts().to_dict(),'history':history})
        print(f'{tag}: test_accuracy={acc:.4f} test_macro_f1={macro:.4f} (val_macro_f1={val_f1:.4f})')
        pooled_true.extend(y_true); pooled_pred.extend(y_pred)
        if val_f1>best_val:
            best_val=val_f1; best_model_state=copy.deepcopy(model.state_dict())

    accs=[f['accuracy'] for f in fold_results]; f1s=[f['macro_f1'] for f in fold_results]
    mean_acc=float(np.mean(accs)); std_acc=float(np.std(accs))
    mean_f1=float(np.mean(f1s)); std_f1=float(np.std(f1s))
    pooled_acc=accuracy_score(pooled_true,pooled_pred)
    pooled_macro=f1_score(pooled_true,pooled_pred,labels=list(range(len(classes))),average='macro',zero_division=0)
    report=classification_report(pooled_true,pooled_pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)

    # Final deployment model: retrain from scratch on all videos (a small set of
    # videos is held out only for epoch selection, never for reported metrics).
    n_val_v=max(1,round(len(videos)*0.2))
    final_train, final_val = pick_val_videos(df, classes, n_val_v, args.seed+999, None)
    print(f'Final deployment model: train={len(final_train)} val={len(final_val)} (epoch selection only)')
    final_model, final_hist, final_vf1 = train_fold(final_train, final_val, classes, args, bbox_map, device, tag='final')
    best_model_state={k:v.detach().cpu().clone() for k,v in final_model.state_dict().items()}

    out=Path(args.results_dir); out.mkdir(parents=True, exist_ok=True); Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    if best_model_state is not None:
        torch.save({'state_dict':best_model_state,'classes':classes,'image_size':224,'crop_to_bbox':bool(bbox_map)},args.model_out)
    metrics={'cv_folds':len(fold_results),'test_accuracy':mean_acc,'test_accuracy_std':std_acc,
             'test_macro_f1':mean_f1,'test_macro_f1_std':std_f1,
             'pooled_accuracy':pooled_acc,'pooled_macro_f1':pooled_macro,
             'classes':classes,'classification_report':report,
             'fold_results':[{k:v for k,v in f.items() if k!='history'} for f in fold_results],
             'history':{**{f['fold']:f['history'] for f in fold_results},'final':final_hist},
             'model_selection':'final deployment model trained on all videos; epoch chosen by its validation macro-F1',
             'labeled_frames':len(df),'frames_with_bbox':n_crop,
             'final_model':{'n_train':len(final_train),'n_val':len(final_val),'val_macro_f1':final_vf1,
                            'train_videos':sorted(final_train.source_video.unique().tolist()),
                            'val_videos':sorted(final_val.source_video.unique().tolist())}}
    with open(out/'metrics.json','w') as f: json.dump(metrics,f,indent=2)
    pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
    pd.DataFrame([{k:v for k,v in f.items() if k!='history'} for f in fold_results]).to_csv(out/'cv_folds.csv',index=False)
    cm=confusion_matrix(pooled_true,pooled_pred,labels=list(range(len(classes))))
    fig,ax=plt.subplots(figsize=(max(7,len(classes)),max(5,len(classes)*.7))); im=ax.imshow(cm)
    ax.set_xticks(range(len(classes)),classes,rotation=45,ha='right'); ax.set_yticks(range(len(classes)),classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Pooled out-of-fold confusion matrix (group CV)')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    print(f'CV mean accuracy={mean_acc:.4f} +/- {std_acc:.4f} | mean macro F1={mean_f1:.4f} +/- {std_f1:.4f}')
    print(f'Pooled accuracy={pooled_acc:.4f} | pooled macro F1={pooled_macro:.4f}')
    print(f'Saved model -> {args.model_out}; results -> {out}/')
if __name__=='__main__': main()
