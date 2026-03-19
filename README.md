# Wildfire Smoke Dataset for Early Detection

## Overview
This project focuses on building a curated and labeled dataset for wildfire smoke detection using computer vision workflows. The goal was not to train a final production model, but to create a structured, reusable dataset that could be used to support future machine learning models for early wildfire smoke detection.

The project includes:
- frame extraction from wildfire surveillance videos
- a custom GUI for manual labeling
- structured CSV annotations
- TensorFlow dataset preparation
- preprocessing, split generation, and dataset inspection scripts

## Repository Contents
- `GUI_labeler.py` — Tkinter-based frame labeling tool
- `dataset_creator.py` / `wildfire_smoke_dataset.py` — TensorFlow dataset builder logic
- `TestingWFDataset.py` — dataset validation, preprocessing, and visualization
- `CS431 Wildfire Dataset Report.pdf` — full written report
- sample output charts and example images

## Important Note
The full extracted frame dataset is **not included** in this repository due to size and storage limitations. Instead, this repo includes the code used to create and structure the dataset, along with representative output images, charts, and documentation showing the resulting workflow and dataset characteristics.

## Project Goals
- Create a labeled wildfire smoke image dataset from surveillance video
- Label both:
  - smoke presence (`yes` / `no`)
  - smoke density (`none`, `low`, `medium`, `high`)
- Prepare the data for use in TensorFlow-based machine learning pipelines
- Document challenges involved in real-world smoke detection data

## Pipeline
1. Collect raw wildfire surveillance videos
2. Extract selected frames from each video
3. Review and label frames using a custom GUI
4. Save labels to CSV
5. Organize frames and metadata into a dataset format
6. Split the dataset into train / validation / test sets
7. Preprocess images for downstream ML workflows

## Tools & Technologies
- Python
- Tkinter
- OpenCV
- PIL
- TensorFlow
- TensorFlow Datasets
- Matplotlib
- CSV-based labeling pipeline

## Dataset Summary
The final dataset was built from approximately 1,000 wildfire surveillance videos and filtered down to a smaller set of high-quality labeled images for training and evaluation.

Labels include:
- smoke present vs. no smoke
- smoke density classification:
  - none
  - low
  - medium
  - high

The dataset was prepared for use in TensorFlow and split into:
- training
- validation
- testing

## Example Outputs Included
This repository includes representative outputs such as:
- smoke label distribution charts
- density label distribution charts
- example preprocessed training images

These are included to demonstrate the final dataset structure and output quality without uploading the complete image dataset.

## Challenges
Some of the main challenges in building the dataset included:
- night footage with poor smoke visibility
- weather interference such as fog, clouds, and glare
- camera obstructions such as poles and structures
- ambiguous visuals like haze, dust, and distant smoke
- class imbalance between smoke and non-smoke examples

## What I Learned
- how important data quality is for ML performance
- how to build a manual labeling workflow for messy real-world data
- how to structure image data and labels for TensorFlow pipelines
- how difficult real-world visual classification can be in unconstrained environments
- how preprocessing and filtering directly affect downstream usability

## Future Improvements
- train and evaluate a wildfire smoke detection model on the dataset
- expand the dataset with more balanced non-smoke examples
- improve labeling efficiency with AI-assisted tools
- explore synthetic augmentation to improve class balance
- release a cleaned public-facing version of the dataset

## Author
Jax Hendrickson  
University of Nevada, Reno  
Computer Science & Engineering

## Additional Documentation
For more detail, see:
- `Wildfire Dataset Report.pdf`
- supporting presentation materials and sample outputs included in the repository
