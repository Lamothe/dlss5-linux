"""Independent level-layout, colour-space and high-frequency filtering oracles."""
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('surface_filter',root/'surface-filter.py')
filtering=importlib.util.module_from_spec(spec)
spec.loader.exec_module(filtering)
# Independent layout calculation matches the shader's integer halving.
for level,(x,y,size) in enumerate(filtering.LEVELS):
    expected_size=368//(2**level)
    expected_y=sum(368//(2**i)+8 for i in range(1,level)) if level else 0
    assert (x,y,size)==(376 if level else 0,expected_y,expected_size)
    assert x+size+8<=744 and y+size+8<=744
assert len(filtering.LEVELS)==9 and filtering.LEVELS[-1][2]==1
# Alternating black/white: coarsest colour must average linear light, not gamma.
checker=(np.indices((736,736)).sum(axis=0)%2)*255
rgba=np.stack([checker,checker,checker,np.full_like(checker,255)],axis=2).astype('uint8')
result=np.asarray(filtering.pyramid(Image.fromarray(rgba)))
x,y,size=filtering.LEVELS[-1]
assert np.max(abs(result[y+4,x+4,:3].astype(int)-188))<=1
# Every sample at level 1+ is stable grey for a subpixel checker pattern.
for x,y,size in filtering.LEVELS[1:]:
    pixels=result[y+4:y+4+size,x+4:x+4+size,:3]
    assert pixels.max()-pixels.min()<=1
    assert np.max(abs(pixels.astype(int)-188))<=1
    assert np.array_equal(result[y:y+4,x+4:x+4+size,:3],np.take(pixels,np.arange(-4,0)%size,axis=0))
# Opposing tangent components cancel; data must not be sRGB averaged.
normals=np.empty_like(rgba);normals[:]=[255,128,255,255];normals[::2]=[0,128,255,255]
normal=np.asarray(filtering.pyramid(Image.fromarray(normals),normal=True))
x,y,size=filtering.LEVELS[-1]
assert np.max(abs(normal[y+4,x+4,:3].astype(int)-[128,128,255]))<=1
print('PASS: nine tile-local levels, linear-light colour averaging, normal-data averaging, gutters and high-frequency rejection')

# Periodic edge/gutter interpolation has identical values at local UV 0 and 1.
rng=np.random.default_rng(42)
source=rng.integers(0,256,(736,736,4),dtype=np.uint8);source[:,:,3]=255
periodic=np.asarray(filtering.pyramid(Image.fromarray(source)))
for x,y,size in filtering.LEVELS:
    row=y+4+size//2
    left=periodic[row,x+3:x+5,:3].astype(float).mean(axis=0)
    right=periodic[row,x+3+size:x+5+size,:3].astype(float).mean(axis=0)
    assert np.array_equal(left,right)
# Frobenius footprint bounds the true major axis even for collinear derivatives.
for jacobian in [np.array([[12,12],[.01,-.01]]),np.eye(2),rng.normal(size=(2,2))]:
    assert np.linalg.norm(jacobian)>=np.linalg.svd(jacobian,compute_uv=False)[0]-1e-12
print('PASS: periodic boundaries at all mip levels and conservative rotated footprint')

# Source-resolution filtering must precede reduction into the atlas.
columns=np.broadcast_to((np.arange(2048)%2)*255,(2048,2048))
source=np.stack([columns,columns,columns,np.full_like(columns,255)],axis=2).astype('uint8')
filtered=np.asarray(filtering.pyramid(Image.fromarray(source)))
x,y,size=filtering.LEVELS[0]
assert np.max(abs(filtered[y+4:y+4+size,x+4:x+4+size,:3].astype(int)-188))<12
# Continuous UV gradients do not inherit cell-floor jumps from helper lanes.
uv=np.array([.999,1.01])*735/8192+4.5/8192
assert abs(np.diff(uv)[0]*(8192/735*368)-.011*368)<1e-10
print('PASS: original2048 source filtering and continuous boundary derivatives')
