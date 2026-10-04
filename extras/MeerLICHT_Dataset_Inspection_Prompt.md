# MeerLICHT Dataset Inspection Prompt

I have the MeerLICHT dataset files, especially:

- `MEERLICHT_images.npy`
- the corresponding labels CSV file

I already visualized some samples, but now I want a careful inspection of the actual dataset structure.

Please inspect the files and report **only what is actually present**. Do not guess.

---

## 1. NumPy Array Structure

Please inspect `MEERLICHT_images.npy` and report:

- Exact array shape
- Total number of samples/candidates
- Data type (`dtype`)
- What each dimension represents

For example, determine whether the structure is something like:

```text
(N, C, H, W)
```

or:

```text
(N, H, W, C)
```

where:

- `N` = number of candidates
- `C` = number of channels
- `H`, `W` = image height and width

---

## 2. Candidate Channel Structure

For a single candidate, check whether the data contains separate astronomical cutouts such as:

- New / Science image
- Reference image
- Difference image
- Scorr / Significance image

If multiple channels exist, identify the **exact channel order**.

Example:

```text
Candidate
├── Channel 0 = ?
├── Channel 1 = ?
├── Channel 2 = ?
└── Channel 3 = ?
```

Do not infer the channel order from appearance alone.

If possible, verify it from:

- metadata,
- accompanying code,
- README,
- dataset documentation.

---

## 3. Cutout/Image Size

Report the exact shape of each individual cutout.

For example:

```text
30 × 30
100 × 100
etc.
```

Also report whether every sample has the same dimensions.

---

## 4. Grayscale vs RGB

The visualization I generated appears colored.

Please determine whether:

- the raw data are actual RGB images,
- the data are grayscale channels combined for visualization,
- or the colors are produced only by a plotting colormap / channel-combination method.

This is important because I want to know what the CNN will actually receive.

---

## 5. Pixel Statistics

For the complete dataset, report:

- minimum value
- maximum value
- mean
- standard deviation
- dtype

If there are multiple channels, also report these statistics **per channel**.

Example:

```text
Channel 0:
min:
max:
mean:
std:

Channel 1:
...
```

---

## 6. Label File Inspection

Inspect the labels CSV and report:

- Exact filename
- Column names
- Number of rows
- Exact label column
- Exact label mapping

For example:

```text
0 = Bogus
1 = Real
```

or whatever the dataset actually uses.

Also report:

```text
Total Real:
Total Bogus:
Real percentage:
Bogus percentage:
```

---

## 7. Image/Label Alignment

Verify that:

- number of image samples == number of labels
- labels correspond to the same order as samples in `MEERLICHT_images.npy`

If candidate IDs exist, use them to verify alignment.

---

## 8. Metadata Inspection

Check whether the dataset contains any metadata such as:

- candidate ID
- object ID
- observation ID
- timestamp
- sky coordinates
- telescope exposure
- magnitude
- seeing
- detector information
- source ID

Report all metadata columns if present.

This is especially important because I want to know whether the same astronomical object can appear multiple times.

---

## 9. Data Leakage Risk

Check whether multiple samples belong to the same:

- astronomical object,
- source,
- observation sequence,
- sky location.

If yes, explain whether a random train/validation/test split could create leakage.

Recommend whether the data should instead be split by:

```text
object ID
source ID
observation ID
or another grouping variable
```

---

## 10. Data Quality Checks

Check for:

- NaN values
- infinite values
- all-zero images
- empty images
- corrupted samples
- inconsistent shapes
- duplicate images
- near-duplicate images

Report how many samples are affected by each issue.

---

## 11. Preprocessing

Determine whether the stored data already appear to be:

- normalized,
- standardized,
- clipped,
- background-subtracted,
- scaled,
- transformed.

If possible, inspect accompanying code or repository files to determine the original preprocessing procedure.

Report whether additional preprocessing is required before CNN training.

---

## 12. Visualize Raw Channels Separately

For at least:

- 3 Real candidates
- 3 Bogus candidates

show each raw channel separately.

For example:

```text
Candidate #X — Real

New:
[image]

Reference:
[image]

Difference:
[image]

Scorr:
[image, if available]
```

Do not combine the channels into RGB for this visualization.

Use grayscale unless the raw data are genuinely RGB.

---

## 13. CNN Input Suitability

Determine whether the dataset can directly support a CNN pipeline like:

```text
Candidate
├── New
├── Reference
└── Difference
        ↓
      CNN
        ↓
Real / Bogus probability
```

If not, explain:

- what channel selection is needed,
- what preprocessing is needed,
- what input shape should be used.

---

## 14. Recommended Model Input

At the end, recommend the most appropriate CNN input format.

For example:

```text
Input shape: (3, 100, 100)

Channels:
0 = New
1 = Reference
2 = Difference
```

or whatever is actually supported by the dataset.

---

## 15. Do Not Train Yet

Do **not** train any model.

This task is only for:

- dataset inspection,
- channel identification,
- label verification,
- data-quality checking,
- preprocessing recommendations.

---

# Final Summary Format

Please end with exactly this compact summary:

```text
Dataset name:
Total samples:
Real samples:
Bogus samples:
Class ratio:

Array shape:
Single sample shape:
Number of channels:
Channel order:
Image size:

Data type:
Global value range:
Per-channel statistics:

Label file:
Label column:
Label mapping:

Metadata available:
Repeated-object risk:
Leakage risk:

NaNs:
Infinite values:
Zero images:
Duplicates:
Other issues:

Already normalized/preprocessed:
Recommended preprocessing:

Recommended CNN input shape:
Recommended channels:

Suitable for Tiny/Medium/Large CNN training:
Yes/No

Major concerns:
```
