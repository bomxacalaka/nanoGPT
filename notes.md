conda activate aiplayground
python data/synthetic_gen/gen.py && python data/shakespeare_char/prepare.py && python train.py config/train_shakespeare_char.py
python sample.py --start="<|start|>" --num_samples=5 --max_new_tokens=1000 --out_dir=out-shakespeare-charw