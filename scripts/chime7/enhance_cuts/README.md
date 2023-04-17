
This is a drop-in replacement for `gss enhance cuts`.

### Setup

To setup

1) Install NeMo as usual

2) Setup GSS -- this is used for the baseline implementation and dataloaders

```
pip install 'cupy-cuda11x<12' # 12 removed cp.bool and had other changes breaking gss

pip install git+http://github.com/desh2608/gss

# this may be necessary if you get numpy errors
pip install numpy==1.21
```

### Example

Check `run_enhance_cuts.sh`.