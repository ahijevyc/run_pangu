import pygrib

fh = pygrib.open('hres_analysis_2025020100.grib2')
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
fields = fh.select(name="Specific humidity", typeOfLevel='isobaricInhPa', level=levels)

for level in levels:
for f in fields:
    print(f.level)


for n in range(len(levels)):
    this_field = fh.select(name='Specific humidity', typeOfLevel='isobaricInhPa', level=levels[n])[0]
    print(this_field.level)
