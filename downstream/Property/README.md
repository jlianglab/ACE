# Property evaluations

## Learned properties

The learned properties refer to the capabilities that ACE acquires through our carefully designed learning objectives and loss functions. These properties are cultivated through targeted self-supervision, enabling the model to capture anatomical structures, spatial consistency, and compositional semantics without relying on manual annotations. 

### Learned property 1: ACE-v2 learned anatomical consistency for the overlapped patches.

```
python test_consistency.py --image_dir IMAGE_PATH --model_path PRETRAINED_WEIGHT_PATH
```

<p align="center"><img width=85% alt="FrontCover" src="images/Learned consistency.png"></p>


### Learned property 2: ACE-v2 learned anatomical uniqueness.

We investigate the learned uniqueness property of ACE-v2, which reflects its ability to balance anatomical diversity and harmony while capturing hierarchical relationships.

```
python test_unique_tsne.py
```
<p align="center"><img width=60% alt="FrontCover" src="images/Learned unique.png"></p>


### Learned property 3: ACE-v2 enhanced feature compositionality

We investigate ACE-v2's ability to preserve the compositionality of anatomical structures in its learned embedding space.

```
python test_KDE.py
```
<p align="center"><img width=60% alt="FrontCover" src="images/Learned composition.png"></p>



## Emergent properties

We consider the following properties **emergent** as ACE-v2 is never trained with losses across patients, but such inter-image consistency below has automatically emerged from training on intra-image losses.


### Emergent property 1: ACE-v2 provides distinctive anatomical embeddings

We investigate ACE-v2's ability to reflect the locality of anatomical structures in its learned embedding space.


```
python tsne_v2.py
```

<p align="center"><img width=60% alt="FrontCover" src="images/Emergent locality.png"></p>


### Emergent property 2: ACE-v2 provides unsupervised cross-patient anatomy correspondence.

To demonstrate the efficacy of our ACE-v2 in capturing a diverse range of anatomical structures, we utilize patch-level features to query the same anatomy across different patients in a zero-shot setting.

```
python correspondence_100image_local.py
```
<p align="center"><img width=60% alt="FrontCover" src="images/Emergent correspondence.png"></p>


### Emergent property 3: ACE-v2 understand anatomical symmetry.

We investigate whether ACE-v2 can capture the intrinsic symmetry of chest X-ray images within its learned embedding space. 

```
python tsne_symmetry.py
```
<p align="center"><img width=50% alt="FrontCover" src="images/Emergent symmetry.png"></p>


### Emergent property 4: ACE-v2 enabled meaningful embedding interpolation and extrapolation.

We investigate ACE-v2’s interpolation/extrapolation capability by computing embeddings from two randomly selected points, and measuring their consistency with the ground truth embeddings.

```
python interpolation.py
```
<p align="center"><img width=50% alt="FrontCover" src="images/Emergent interpolation.png"></p>
