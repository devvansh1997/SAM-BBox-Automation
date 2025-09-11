# Setup and Installation

```
pip install -r requirements.txt
```

# Running Eval
```
python evaluate.py --image-dir "path/to/camvid/val" --mask-dir "path/to/camvid/val_labels"
```
or
```
python evaluate.py --image-dir "path/to/camvid/val" --mask-dir "path/to/camvid/val_labels" --limit 15
```
# Running Visualization
```
python visualize.py --image-dir "path/to/camvid/val" --mask-dir "path/to/camvid/val_labels"
```