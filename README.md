## Dataset Download

The dataset for this project is hosted on Kaggle. You can download it using the following link:

[Bird Species Classification Dataset](https://www.kaggle.com/datasets/anupriyavm/bird-species-classification)

### Download using Kaggle API
To download the dataset via the Kaggle API, run:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("anupriyavm/bird-species-classification")

print("Path to dataset files:", path)
