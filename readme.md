Deberas tener poetry

Deberas tener python 3.9.23 que poetry pueda apuntar

#readmeta.py usage 

```bash
python readmeta.py --path ./input/metadata.csv 
```


```bash
python readmeta.py --path ./input/metadata.csv --separator ";"
```


```bash
python readmeta.py --path ./input/metadata.csv --drop-missing-coords

```


##Usage of split_by_dir.py
#Tirar filas malas:
```bash
python split_by_dir.py \
  --csv .../metadata.csv \
  --separator ';' \
  --output-dir .../output \
  --max-per-class 0 \
  --keep-first-duplicate \
  --drop-out-of-bounds
```

#Ajustar al borde:
```bash
python split_by_dir.py \
  --csv .../metadata.csv \
  --separator ';' \
  --output-dir .../output \
  --max-per-class 0 \
  --keep-first-duplicate \
  --clamp-out-of-bounds
```