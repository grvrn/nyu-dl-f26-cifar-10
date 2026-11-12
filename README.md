# CIFAR-10: ResNets Under Five Million Parameters

This repository contains the four approaches discussed in the project report: a supervised ResNet baseline, an AutoAugment ResNet, plain-ResNet knowledge distillation, and WideResNet-to-SE-ResNet distillation. All submitted networks have fewer than five million trainable parameters. Teachers are training aids and are not used in submission inference.

## Models

| Notebook | Submitted network | Parameters | Training configuration |
| --- | --- | ---: | --- |
| [cifar10_resnet_5m.ipynb](cifar10_resnet_5m.ipynb) | ResNet, blocks `[3,3,3]` | 4,327,754 | 150 epochs; stratified 80/20 split; crop and flip |
| [02_autoaugment_resnet.ipynb](02_autoaugment_resnet.ipynb) | ResNet, blocks `[4,4,3]` | 4,697,162 | 200 epochs; random 98/2 split; AutoAugment, erasing, label smoothing |
| [03_distillation_resnet.ipynb](03_distillation_resnet.ipynb) | ResNet, blocks `[4,4,3]` | 4,697,162 | ResNet teacher for 160 epochs, student for 250; random 98/2 split; distillation and MixUp/CutMix |
| [wrn_distillation.ipynb](wrn_distillation.ipynb) | SE-ResNet, blocks `[4,4,3]` | 4,733,610 | Reuse/train WideResNet teacher; final student uses all labeled images and a previously selected stopping epoch |

All four submitted networks use stage widths `[64,128,256]`. The current WRN notebook implements **final full-data student training**, not the earlier 80/20 student experiment. Its final student therefore has no held-out validation accuracy. Validation results across different splits are not directly comparable.

## Environment and competition data

The supported workflow is Kaggle Notebooks with a CUDA GPU. Import the desired `.ipynb`, attach the official competition data, and run its cells in order. The notebooks do not automatically distribute computation across two GPUs; a single GPU is sufficient. The baseline has a CPU fallback, but CPU training will be slow. The other three notebooks require CUDA.

Dependencies are Python, PyTorch, torchvision, NumPy, Pillow, and, for `02` and `03`, pandas. Use a Kaggle Python image with compatible PyTorch/torchvision versions supporting `torch.amp.autocast` and `torch.amp.GradScaler`. Package versions are not pinned in this repository, so preserve the environment details of any reported run.

The baseline and WRN notebooks expect this competition location by default:

```text
/kaggle/input/competitions/cifar-10-res-net-architecture-challenge-dl/
├── train_labels.csv
├── train_images/
├── test.csv
├── test_images/
└── sample_submission.csv
```

If Kaggle mounts the data elsewhere, change `INPUT_ROOT` in their configuration cell.

Input handling differs between the notebook families:

- **Baseline and WRN:** `train_labels.csv` must contain `id,label`. Training IDs resolve to filenames such as `train_00001.png`. Labels can be the ten numeric class strings or the standard CIFAR-10 names; the submission code maps classifier positions to official indices. Test IDs are taken from `sample_submission.csv`, with numeric, `test_00001`, and `test_00001.png` forms supported. The exact stratified split requires each class count to be divisible by five.
- **02 and 03:** both CSVs must contain a `filename` column matching files inside their image directories. Training labels must be integers from 0 to 9, and `test.csv` must contain numeric `id` values matching the sample submission. These notebooks search `/kaggle/input` for the CSVs and train/test image directories and preload images into RAM. Attach only the intended competition dataset to avoid ambiguous automatic discovery.

The competition submission must be named `submission.csv`, contain exactly `id,label`, preserve sample submission ordering, and provide one integer label from 0 to 9 for every test image.

## 1. Supervised baseline

Open `cifar10_resnet_5m.ipynb` and run top to bottom. The notebook creates the fixed stratified split, trains for 150 total epochs, restores the checkpoint with the highest validation accuracy, and generates single-view test predictions.

Outputs in `/kaggle/working`:

| File | Purpose |
| --- | --- |
| `best_resnet.pt` | Best validation model and prediction metadata |
| `last_checkpoint.pt` | Full resumable training state, including best weights |
| `history.csv` | Per-epoch training/validation metrics |
| `split.csv` | Training/validation assignment for every labeled image |
| `submission.csv` | Test predictions |

### Resume the baseline

With `AUTO_RESUME=True`, rerunning uses `/kaggle/working/last_checkpoint.pt` if present. Otherwise attach the saved checkpoint as input and set:

```python
RESUME_CHECKPOINT = Path('/kaggle/input/your-saved-output/last_checkpoint.pt')
```

Keep the original configuration and data unchanged. The next run starts after the last completed epoch and stops at epoch 150. A partially completed epoch repeats. Set `AUTO_RESUME=False` only to intentionally start over; the new run replaces checkpoint outputs.

For prediction only, make `best_resnet.pt` available in `/kaggle/working`, run the import/configuration and model-definition cells, and then run **Generate the Kaggle submission**. The prediction cell reconstructs the model and preprocessing from the checkpoint metadata.

## 2. AutoAugment ResNet

Open `02_autoaugment_resnet.ipynb`, attach competition data with the `filename` columns described above, enable a GPU, and run all cells.

The notebook:

1. Preloads images and makes a seed-42 random 98/2 split.
2. Trains a `[4,4,3]` ResNet for 200 epochs using crop, flip, AutoAugment, random erasing, and label smoothing.
3. Restores the best validation weights and saves `aug_443.pt`.
4. Combines logits from original and horizontally flipped test images, then writes `submission.csv`.

Outputs are relative to the notebook's working directory, normally `/kaggle/working` on Kaggle. Validation progress prints every five epochs and at the final epoch.

**Interruption behavior:** best weights are held in memory until training finishes. This notebook has no automatic training resume or per-epoch checkpoint file. For an unattended run, use Kaggle's background **Save & Run All** workflow and preserve the resulting outputs.

To predict from an existing `aug_443.pt`, run setup/data/model definitions, construct `ResNet([4,4,3], [64,128,256]).to(DEVICE)`, load the checkpoint's `state_dict`, assign it to `model`, and run the final prediction cell instead of the training cell.

## 3. Plain-ResNet distillation

Open `03_distillation_resnet.ipynb`, attach the same data format as `02`, enable a GPU, and run all cells.

The notebook trains a larger ResNet teacher with blocks `[4,4,4]` and channels `[96,192,384]` for 160 epochs. It restores and freezes the best teacher before training the student for 250 epochs. Student training uses a temperature of 4, a distillation weight of 0.7, and MixUp/CutMix. Both phases use the same random 98/2 split.

Outputs:

| File | Purpose |
| --- | --- |
| `teacher.pt` | Best plain-ResNet teacher weights |
| `distilled_student.pt` | Best student weights, validation accuracy, and parameter count |
| `submission.csv` | Student predictions using original/flip logit aggregation |

Progress prints every ten epochs and at the last epoch. Printed epoch indices start at zero: `ep 159` means 160 epochs completed.

**Interruption behavior:** neither phase has automatic resume. A top-to-bottom run retrains the teacher even if `teacher.pt` is present. Model files are saved after each training phase completes.

For student-only prediction, run setup/data/model definitions, construct `ResNet([4,4,3], [64,128,256]).to(DEVICE)`, load `distilled_student.pt`'s `state_dict` into it, assign it to `student`, and run the final prediction cell. Teacher weights are unnecessary for inference.

## 4. WideResNet-to-SE-ResNet final training

Open `wrn_distillation.ipynb`. This notebook reuses the selected teacher and trains a **fresh student on 100% of labeled images**. Earlier student weights are used neither as initialization nor as training targets.

### Required inputs for the reported final run

Attach:

1. The official competition data.
2. Saved notebook outputs containing **`wrn_teacher.pt`**.
3. One earlier 80/20 student selection file: **`wrn_student_best.pt`**, **`wrn_student_last_checkpoint.pt`**, or **`wrn_student_history.csv`**. This determines the stopping epoch only.

Alternatively, set the selected one-based epoch explicitly. For the reported 192-epoch final run, use:

```python
FULL_STUDENT_EPOCHS = 192
```

With that setting, no earlier student selection file is needed. Keep `STU['epochs']=200`: this is the original learning-rate schedule horizon, not the final stopping epoch. To resolve multiple attached selection files, set `PRIOR_STUDENT_SELECTION_FILE` to the intended file's exact path.

### Teacher reuse and run order

Run the notebook top to bottom:

1. Setup reconstructs the teacher's stratified 80/20 split and class mapping.
2. Phase 1 checks `/kaggle/working/wrn_teacher.pt`, then searches attached inputs under `/kaggle/input`. If found, it loads and freezes the teacher without training it. Saved class/split metadata is checked when available.
3. If no exported teacher exists, it trains the WideResNet-28-8 teacher for 150 epochs, using a teacher resume checkpoint when available. This fallback does not require an earlier teacher run, but the final student still needs a selected epoch or explicit `FULL_STUDENT_EPOCHS`.
4. Phase 2 builds a fresh SE-ResNet and trains on all labeled images through the selected epoch. No validation score or best-validation selection is computed in this phase.
5. The final cell reloads the exported full-data student, performs ten-view inference, validates IDs/labels/coverage, and writes `submission.csv`.

The teacher remains frozen throughout student training. The earlier 97.31% student validation result belongs to the separate 80/20 experiment, not this full-data model. The current notebook does not automatically reproduce that earlier student selection experiment.

### WRN outputs and resume

| File | Purpose |
| --- | --- |
| `wrn_teacher.pt` | Teacher export, also written when reusing an attached teacher |
| `wrn_teacher_last_checkpoint.pt` | Full teacher resume state, when teacher training runs |
| `wrn_teacher_best.pt` | Best teacher checkpoint, when teacher training runs |
| `wrn_teacher_history.csv` | Teacher metrics, when teacher training runs |
| `wrn_student_full_last_checkpoint.pt` | Full-data student resume state after each completed epoch |
| `wrn_student_full_history.csv` | Full-data student losses and epoch timing |
| `wrn_student_full.pt` | Final student weights and inference metadata |
| `split.csv` | Original 80/20 assignment retained for teacher checks |
| `submission.csv` | Final student predictions |

For a future full-data student resume, attach the same teacher and `wrn_student_full_last_checkpoint.pt`. With `AUTO_RESUME=True`, the notebook discovers that exact filename in working storage or attached inputs. It also recovers the stopping epoch from it, so the earlier 80/20 selection file is no longer needed. For an explicit path:

```python
STUDENT_FULL_RESUME_CHECKPOINT = Path(
    '/kaggle/input/your-saved-output/wrn_student_full_last_checkpoint.pt'
)
```

Do not point this setting at `wrn_student_last_checkpoint.pt`: that belongs to the earlier 80/20 experiment. Resume checks require the same data, configuration, and teacher weights. If teacher training itself was interrupted before export, set `TEACHER_RESUME_CHECKPOINT` to its full teacher checkpoint.

For prediction only, make `wrn_student_full.pt` available in `/kaggle/working`, run the import/configuration and model-definition cells, then run the final submission cell. The final student does not need the teacher at inference.

## Saving runs and submitting predictions

- Use a separate Kaggle notebook/run for each approach, since all write `submission.csv`.
- For long unattended runs, launch **Save Version → Save & Run All**. This executes a fresh copy from the top and remains subject to Kaggle runtime and quota limits. It resumes only when the required checkpoint is accessible to the new run.
- Preserve saved notebook outputs, download checkpoints, or enable file persistence where available. Writing a checkpoint into `/kaggle/working` alone does not guarantee it will survive session termination.
- Attach previous saved notebook outputs as inputs to reuse checkpoints. Checkpoint files are not included in this repository.
- Download the desired run's `submission.csv` and submit it through the official Kaggle competition page. CSV generation does not submit automatically or compute hidden-test accuracy.

The baseline and WRN submission cells explicitly check the schema, sample order, class mapping, duplicate IDs, and test-image coverage. The older `02` and `03` cells follow the sample ordering but have fewer checks; verify exactly two columns (`id,label`), complete row coverage, and integer labels 0–9 before uploading.

## Additional notebook

[cifar10_resnet_teacher.ipynb](cifar10_resnet_teacher.ipynb) is a separate larger-teacher experiment. It is not the WideResNet teacher used by `wrn_distillation.ipynb`, and its `best_teacher.pt` is not interchangeable with `wrn_teacher.pt`.
