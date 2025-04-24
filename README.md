
# GPT-2 Model Training

## Project Overview
This project implements the training of a GPT-2 model using the Hugging Face `transformers` library. The training process involves loading a custom dataset, tokenizing the text, training the model, and evaluating its performance. The project also includes text generation capabilities after the model is trained.

## Setup & Run Instructions

### 1. **Clone the repository**
   ```bash
   git clone git@github.com:littlemoon0210/LLM4EDA_pj1.git
   cd LLM4EDA_pj1
   ```

### 2. **Install dependencies**
   Ensure Python is installed, then install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Download dataset

To begin using this project, you need to download the necessary datasets. You can do this by running the provided Python script, which will automatically fetch and save the datasets in the `data` folder.

#### Steps:
1. Clone or download the repository to your local machine.
2. Run the following Python script:

    ```bash
    python download_dataset.py
    ```

This will start downloading the datasets, and the progress will be displayed. The script fetches the following datasets:

- webtext
- small-117M
- medium-345M
- large-762M
- xl-1542M

Each dataset is split into the following parts: `train`, `valid`, and `test`. The downloaded files will be stored as `.jsonl` files in the `data` folder.

For more details, please refer to the following link: [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset.git)

If you prefer, you can also manually download the datasets from the provided links in the script, but the automatic download method is recommended for ease of use.



## Usage

### 1. **Run the Training Script**
   To start the training process, execute the `train.py` script:
   ```bash
   python train.py
   ```
   This will load the dataset, initialize the GPT-2 model, and begin training. Follow the prompts to monitor training progress and adjust hyperparameters if needed.

### 2. **Training Process**
   - The `train.py` script loads and preprocesses the dataset using the `dataset.py` script, tokenizes the text, and prepares it for training.
   - The `model.py` script defines the GPT-2 model architecture and configures the necessary parameters (e.g., number of layers, attention heads).
   - The training is tracked with Weights and Biases (WandB) for real-time visualization.
   - The model will be saved at the end of training in the `./logs/final_model` directory.

### 3. **Generate Text after Training**
   After training, you can use the `generate.py` script to generate text using the trained model:
   ```bash
   python generate.py --model_path ./logs/final_model --length 100
   ```
   This will generate a sequence of 100 tokens based on the trained model.

### 4. **Resuming Training (Optional)**
   If you need to resume training from a checkpoint, the script will automatically check for the latest checkpoint in the `./logs/checkpoint-last` directory and continue training from that point.
