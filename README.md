The implementation of the Transformer model in this project is based on the techniques described in [this Medium article](https://medium.com/@khanarsh0124/gsoc-2024-with-humanai-text-recognition-with-transformer-models-de86522cdc17) by Arsh Khan (2024).

# Problem Statement
To build a model based on convolutional-recurrent, transformer, or self-supervised architectures for optically recognizing the text of each data source. THe model should be able to detect the main text in each page, while disregarding other embellishments. Pick the most appropriate approach and discuss the strategy.

# Dataset
The dataset consists of 6 scanned early modern printed sources. The images have a simple recognition applied that reflects the limitations of the OCR already used (missed letters, incorrectly recognized words...), each source is saved as separate PDF file.<br>

The dataset also includes a transcription of the first 3 pages of each PDF source – they should be used as reference while training the AI models for the project.

# Approach

## Data Preprocessing

### 1. Conversion to PNG Image
The Pdf files are processed and each page is stored as a PNG file.<br>
Denoising and Thresholding:
```python
gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
```
Line Removal:
```python
# Vertical Lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# Horizontal Lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
```
Image Repair and Enhancement:
```python
repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
removed = 255 - removed
dilate = cv2.dilate(removed, repair_kernel, iterations=5)
```

![](assets/Preprocess_img.png)

Then we employ a custom trained U-Net Segmentation Model modified with attention mechanism to identify the text region and crop the images by identifing the contour regions<br>

![](assets/u_net_seg_ex.png)

### 2. Line Segmentation
- We convert the image to binary format with text being white and background black
- Calculate the Horizontal Projection Profile(HPP)
- Set a threshold value to identify peak regions corresponfing to text lines
```python
threshold = (np.max(hpp)-np.min(hpp))/2
```
![](assets/hpp_img.png)

- We use A* Path Finding Algorithm to identify text lines and segment the image

![](assets/seg_img.png)

## Data Augmentations

```python
augmentations = [
        A.Rotate(limit=3, p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ElasticTransform(alpha=0.3, sigma=100.0, p=1.0),
        A.OpticalDistortion(distort_limit=0.03, shift_limit=0.03, p=1.0),
        A.CLAHE(clip_limit=2, tile_grid_size=(4, 4), p=1.0),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), shear=(-2, 2), p=1.0),
        A.Perspective(scale=(0.01, 0.03), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GridDistortion(num_steps=3, distort_limit=0.02, p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0),
        A.MedianBlur(blur_limit=(3, 5), p=1.0),  # Ensures odd kernel size
    ]
```

## Vision Transformers (ViT)
