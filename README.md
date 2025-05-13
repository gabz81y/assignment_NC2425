# Neural Computing: Food Classification with CNN 


## Project Description
This project implements a Convolutional Neural Network (CNN) for multi-class classification of food images across 91 distinct categories. The dataset is split into training (~44k samples) and testing (~22k samples) subsets and undergoes preprocessing. A custom model is then built using TensorFlow, trained on the dataset, and evaluated for accuracy. The main challenge involved tuning hyperparameters—such as learning rate and dropout—and shaping an effective CNN architecture, with fixed random seeds used to ensure reproducibility. Finally, a simulation is performed in which a hypothetical user submits 10 random food images to be classified by the trained model.

The purpose of this project is to demonstrate the application of deep learning for image classification in a real-world-like setting such as food recommendation systems.

It is designed for the NC2425 Neural Computing course at Leiden University and runs on LIACS remote servers.

### Features
- Classifies images across 91 food categories.
Classifies food images into 91 distinct categories.
- Builds and trains a Convolutional Neural Network (CNN) from scratch, without relying on pretrained models.
- Simulates real-life usage by testing with random food images.
- Tracks training progress using accuracy and loss per epoch.
- Integrates visualization of the model development process for interpretation and debugging.
-Designed for deployment on LIACS remote servers via SSH, integrated with each student’s /data directory.


## Prerequisites
This project must be run on LIACS remote servers and is intended only for Leiden University students who have:

- Valid Leiden University account

- SSH access to ssh.liacs.nl and internal servers (e.g., vibranium)

- Working WSL terminal 

- Basic knowledge of SSH, Git, and Python venv

- Required Python packages from requirements.txt

## Running Instructions
Setup Instructions (via LIACS SSH)

___

1. Open WSL terminal.

___

2. SSH into the LIACS login server:

```bash
ssh s[student_number]@ssh.liacs.nl
```


(Enter your Brightspace/ULCN password when prompted)

(During the first time, write 'yes' when asked to connect with a fingerprint.)

___

3. SSH into the Internal Server

```bash
ssh vibranium
```

(Enter your Brightspace/ULCN password when prompted)

(During the first time, write 'yes' when asked to connect with a fingerprint.)
___
4. Start a screen session:  

During the setup
```bash
screen -S cnn_training
```
During the next sessions to restore the screen
During the setup
```bash
screen -r cnn_training
```
[TIP - use whenever in screen you want to reset the content in the screen]
```bash
clear
```
___
5. Navigate to your home directory

```bash
cd /home/s[student_number]/
```
___
6. Check if the /data/s[student_number] directory exists

```bash
cd /data/s[student_number]
```

If it does not exist, create it:
    
```bash
mkdir /data/s[student_number]
cd /data/s[student_number]
```
___
7. Clone the repository: 
```bash
git clone https://github.com/gabz81y/assignment_NC2425.git
cd assignment_NC2425
```
___
8. Sync with Latest Changes:
```bash
git pull origin main --rebase
```
___
9. Set Up a Python Virtual Environment
```bash
python -m venv nc_venv
source nc_venv/bin/activate
```
___
10. Install Dependencies
```bash
pip install -r requirements.txt
```
___
11. Download and Prepare the Dataset
```bash
python get_data.py
```
Verify that train and test directories have been created:
```bash
ls
```
___
12. Convert Notebook to Script
```bash
jupyter nbconvert --to script assignment_NC2425.ipynb
```
___
13. Run the CNN Training Script
```bash
python assignment_NC2425.ipy
```
___
14. Detach from the screen session (only during the setup session) 
```bash
Ctrl + A, then D
```
___


## Usage
This CNN-based food classifier supports real-world applications such as:

- Restaurant Recommendation Systems – Recommend dishes based on identified food types.

- Diet Tracking Apps – Automatically log and categorize meals from photos.

- Kitchen Assistants – Detect food in real time from images. 

## Authors 
GROUP_10: NC_PA_2425 10 - NC_PA_2425_311543_39802_10
- **Czapska Gabriela s4053672** 
- **Doupovcová Livia s3952320** 
- **Hanganu Ioana s3792773** 
- **Snepvangers Alex s3700216**
## Licence
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
