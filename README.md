# Nikola Hu's ARC-Prize-2024 Submission Snapshot

This repository contains my final submission for the [ARC-Prize-2024 competition](https://www.kaggle.com/competitions/arc-prize-2024), which achieved 31 points (15-16th place in this largest Kaggle competition). The snapshot was taken shortly after the submission deadline. This repository combines two local git repositories into one and includes intermediate data downloading script to facilitate reproduction and improvement of the results. I kept the word "folder" in the subdirectory names to indicate they were separate git repositories.

## Key Features & Areas for Improvement

Several interesting approaches were explored in this project that could be further investigated:

- **Active Inference Implementation**: MindAI/Jack Cole's team shared it in June (I discovered it when I got stuck at 4 points)
- **Reverse Augmentation**: Originally implemented as "consistency" - later identified as similar to MindAI's Reverse Augmentation mentioned in June
- **Custom Positional Encoding**: Outperforms traditional NLP positional encoding (implemented as `grid_encoding` in the code)
- **Transformer Mask Hack**: (Disabled in final submission due to negative performance impact)
- **Progressive Head**: Impact unverified due to later-discovered unrelated bugs in original experiments
- **Final Ensemble**: Ensemble approach used in submission

## Prerequisites

- Python 3.11.9
- PyTorch 2.2.1
- Other common libraries

## Repository Structure

The repository is organized into two main folders:
- `dev_folder`: Development and training code
- `submission_folder`: Final submission code

### Training Environment (dev_folder)

#### Training the Model
```bash
cd dev_folder
bash scripts/download_intermediate_data_directly.sh
bash scripts/train_using_barc_data.sh
```

#### Monitoring Training Progress
Start TensorBoard to visualize training metrics:
```bash
cd dev_folder
bash scripts/start_tensorboard.sh --runs
```
Access the TensorBoard interface at `http://localhost:6006`

### Submission Environment (submission_folder)

Submission Environment is to simulate and test the kaggle submission.

To run the submission code:
```bash
cd submission_folder
bash download_model_from_kaggle.sh
cd working  # submission_folder/working
python ensemble.py
```

## Known Issues

The checkpoint loading probably did not restore the optimizer's state correctly. Loading from a checkpoint performs differently from the original training trajectory.

## Remaining Questions

### Core Research Questions
1. Can this start-from-scratch transformer-based solution reach the top of the leaderboard?
2. Is it possible to reach the 85% grand prize threshold with this approach?

### Additional Hypothesis
- A general language model is not necessary for this task

## License

MIT License

## Contact

Please refer to the commit email for contact information.

## Acknowledgments

- ARC-Prize-2024 competition organizers
- Mehran Kazeminia for the comprehensive notebook containing previous winning solutions: [Kaggle Notebook](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions)
- Previous winners including icecuber and others whose solutions provided valuable insights
- MindAI's interview: [YouTube](https://www.youtube.com/watch?v=jSAT_RuJ_Cg). My original model performed @ 30-40% for the public evaluation set and scored only 4 points on the private test set. It was a very dark time for me until I discovered this interview!
- Re-arc dataset from [michaelhodel/re-arc](https://github.com/michaelhodel/re-arc)
- BARC dataset from [xu3kev/BARC](https://github.com/xu3kev/BARC.git)

## Background Context

This project was developed during the final month of the competition, focusing on a transformer-based approach rather than a general language model. The repository provides a foundation for further research and improvement of the proposed solutions.

### A note regarding this start-from-scratch transformer model

While using an existing model might have been more straightforward, this project intentionally started from scratch to conduct fundamental research on transformer architectures. This approach allowed for experimentation with:
- Custom masking mechanisms
- Novel grid encoding techniques
- Progressive head architectures
- Various other architectural experiments

Unfortunately, not being able to use vLLM to accelerate the transformer was a significant limitation, which slowed down the final inference time.