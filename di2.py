import glob

gf = glob.glob("output/nadesico_compose_*.csv")
result = open("output/ssim_nadesico.csv", "w")
res = []
for f in gf:
    ff = open(f)
    result.write(ff.read())
    ff.close()

result.close()
