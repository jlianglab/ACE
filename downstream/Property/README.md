# Property evaluations

## Learned properties

The learned properties refer to the capabilities that ACE acquires through our carefully designed learning objectives and loss functions. These properties are cultivated through targeted self-supervision, enabling the model to capture anatomical structures, spatial consistency, and compositional semantics without relying on manual annotations. 

### Learned property 1: ACE-v2 learned anatomical consistency for the overlapped patches.

```
python test_consistency.py --image_dir IMAGE_PATH --model_path PRETRAINED_WEIGHT_PATH
```

<p align="center"><img width=100% alt="FrontCover" src="images/Learned consistency.png"></p>


### Learned property 2: ACE-v2 learned anatomical uniqueness.

```
python test_unique_tsne.py
```
<p align="center"><img width=100% alt="FrontCover" src="images/Learned unique.png"></p>
