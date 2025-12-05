
# Coins Segmentation and Value Counting

This repository contains a solution for the **Soft Computing 2025/26** task focused on **image segmentation and coin value estimation**.

---

## Project Description
The goal is to **segment coins from images** and calculate the **total value of coins** present in each image using predefined heuristics.

### Coin Value Heuristics
- **Gold coin** → `1`
- **Red coin** → `2`
- **Star coin** → `5`

**Example:**  
If an image contains one gold, one red, and one star coin:  
`1 + 2 + 5 = 8`

---

## Dataset
- Images are located in the `data` folder.
- Ground truth values for each image are stored in `coin_value_count.csv`.

---

## Task Requirements
- Detect and segment coins in each image.
- Calculate the total value of coins per image.
- A coin is considered **present** if:
  - It is not mostly cropped at the image edges.
  - It is clearly visible without zooming.
- Compute **Mean Absolute Error (MAE)** between predicted and actual values.
- The solution should:
  - Apply the same preprocessing and operations to all images.
  - Achieve **MAE < 3** 

---
