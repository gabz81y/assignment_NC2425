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

1. Open WSL Terminal

Launch your terminal using Windows Subsystem for Linux (WSL).

___

2. Connect to the LIACS Login Server

SSH into the LIACS login node using your student credentials:
```bash
ssh s[student_number]@ssh.liacs.nl
```
On your first attempt, you'll be prompted to confirm the server’s authenticity. Type "yes" when asked to save the host key.

Enter your **Brightspace/ULCN** password when prompted.

___

3. SSH into the Internal Server (VIbranium)

Once connected to the login node, SSH into the internal server:
```bash
ssh vibranium
```
If you receive a warning about a changed host key (potential MITM attack), clear the old key and reconnect:

```bash
ssh-keygen -R vibranium
ssh vibranium 
```
Accept the host key again if prompted and enter your password.

___

4. (Optional) Start a screen Session

Using screen allows your training jobs to continue running even if your SSH session disconnects.

To start a session for the first time:
```bash
screen -S cnn_training
```
To reconnect to a session later:
```bash
screen -r cnn_training
```
To clear the terminal content inside a screen session:
```bash
clear
```

___

5. Navigate to Your Home Directory

```bash
cd /home/s[student_number]/
```
___

6. Navigate to or Create the /data Directory

First time only:
```bash
mkdir /data/s[student_number]/
cd /data/s[student_number]
```

Subsequent use:
```bash
cd /data/s[student_number]/
```

___

7. Clone or Access the assignment_NC2425 Repository

During setup:
```bash
git clone https://github.com/gabz81y/assignment_NC2425.git
cd assignment_NC2425
```
Later:
```bash
cd assignment_NC2425
```

___

8. Sync with the Latest Changes:

During setup:
```bash
git pull origin main --rebase
```

Later:
```bash
git pull 
```

___

9. Access the Python Virtual Environment

During setup:
```bash
python -m venv nc_venv
source nc_venv/bin/activate  
```

Later:
```bash
source nc_venv/bin/activate  
```

___

10. Install Dependencies (only during setup)
```bash
pip install -r requirements.txt
```

___

11. Download and Prepare the Dataset (setup only)
```bash
python get_data.py
```

Once complete, verify the dataset was extracted properly:
```bash
ls
```
You should see both "train" and "test" directories in the current folder.

___

12. Convert the Notebook to a Python Script (setup only)

Convert the Jupyter notebook to a .py script using:
```bash
jupyter nbconvert --to script assignment_NC2425.ipynb
```
This will generate assignment_NC2425.py in the same directory.

___

13. Run the CNN Training Script

To start model training, run:
```bash
python assignment_NC2425.py
```

___

14. Detach from the Screen Session (if using screen)

If you're running inside a screen session and want to safely detach:
```bash
Ctrl + A, then D
```
This will keep the training running in the background, allowing you to reconnect later.

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
