f1 = open("12.txt")
f2 = open('image/nadesico.txt')
f3 = open('image/nadesico2.txt', "w")
import numpy as np
line = np.array(f1.readlines())
print(line)
nadesico = np.array(f2.readlines())
print(nadesico)
res  = []
f2.close()
f1.close()

print(nadesico.shape)
line2 = []
import tqdm
for i in tqdm.tqdm(range(line.shape[0])):
    r = line[i]
    r = r.replace(".png_", ".png,image/nadesico_small_1080_crop/")
    r = r.replace(".csv", "")
    r = r.replace("gs://topgate-ai-dev-df/output", "image/nadesico_big_crop_ss")
    line2.append(r)
"image/nadesico_big_crop_ss/nadesico_big.frames.000266.png.crop.16.png,image/nadesico_small_1080_crop/nadesico_small_1080.frames.000267.png.crop.16.png"
res = np.setdiff1d(nadesico,line2).astype("string")
"""
res = res[np.where('crop.05' not in res)]
res = res[np.where('crop.11' not in res)]
res = res[np.where('crop.17' not in res)]
res = res[np.where('crop.23' not in res)]
res = res[np.where('crop.24' not in res)]
res = res[np.where('crop.25' not in res)]
res = res[np.where('crop.26' not in res)]
res = res[np.where('crop.27' not in res)]
res = res[np.where('crop.38' not in res)]
res = res[np.where('crop.29' not in res)]
"""
res = res[np.where(np.char.find(res, 'crop.05') == -1)]
res = res[np.where(np.char.find(res, 'crop.11') == -1)]
res = res[np.where(np.char.find(res, 'crop.17') == -1)]
res = res[np.where(np.char.find(res, 'crop.23') == -1)]
res = res[np.where(np.char.find(res, 'crop.24') == -1)]
res = res[np.where(np.char.find(res, 'crop.25') == -1)]
res = res[np.where(np.char.find(res, 'crop.26') == -1)]
res = res[np.where(np.char.find(res, 'crop.27') == -1)]
res = res[np.where(np.char.find(res, 'crop.28') == -1)]
res = res[np.where(np.char.find(res, 'crop.29') == -1)]
print(res)
print(res.shape)
f3.write("\r".join(res))
f3.close

import tarfile
tar = tarfile.open("image/nadesico2.tar", "w")
for name in res:
    r1, r2 = name.strip().split(",")
    if len(r1) > 0:
        tar.add(r1)
    if len(r2) > 0:
        tar.add(r2)
tar.close()
print("image/nadesico2.tar")

