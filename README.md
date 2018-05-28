# GenAttack: Practical Black-box Attacks with Gradient-Free Optimization.

This repo has an implemntation for our paper [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090)

### Instructions
Download `Inception-v3` model checkpoint

```
python setup_inception.py
```

Run attack without dimensionality reduction and adaptive parameter scaling
```
 python main.py --input_dir=./images/ --test_size=1 \
    --eps=0.05 --alpha=0.15 --mutation_rate=0.005  \
    --max_steps=500000 --output_dir=attack_outputs  \
    --pop_size=6 --target=704 --adaptive=False
```

![Attack example with no dimensionality reduction](attack_example_no_dimred.png)
**Original class:** Squirrl, **Adversarial class**: Parking Meter, **Number of queries**=74,171


***For more query efficiency***

Run attack with dimensionality reduction and adaptive parameter scaling

```
python main.py --input_dir=./images/ --test_size=1 \
    --eps=0.05 --alpha=0.15 --mutation_rate=0.10  \
    --max_steps=100000 --output_dir=attack_outputs \
    --pop_size=6 --target=704 --adaptive=True --resize_dim=96
```

![Attack example](attack_example.png)
**Original class:** Squirrl, **Adversarial class**: Parking Meter, **Number of queries**=11,696


**More options**:
* If you want to test on a single image, add the FLAG: `--test_example=xx`.
* To specify a target class, instead of using a random target, add the flag `--target=xx`.

---
## Maintainer:
* This project is maintained by: Moustafa Alzantot [(malzantot)](https://github.com/malzantot)

