# Audio based neural network debugging
Debugging and analyzing neural network training based on audio generated corresponding to gradients during training time.  

Doing this project because I have a lot of time on my hands and it seemed like a good idea to pass my time. :neutral_face:

## Generated Samples:
Most of the generated samples have been on SGD using variable learning rates.  
The training has been done on a four layer NN for 200 steps in one pass using a small batch size(20)  
Higher norms are indicated with higher pitch of the sound. In most cases the pitch increases to indicate the increase in gradient over time.

### LR: 0.01

Audio Sample: [lr_0.01](/training_audio/sgd_0.01.wav)  
Notice the pitch increasing slowly over time indicating slow increase in gradients.

### LR: 0.1

Audio Sample: [lr_0.1](/training_audio/sgd_0.1.wav)  
Notice how the gradients diverge due to higher learning rate and then converge back in again and again as it jumps great distances on the error curve towards minima.  

### LR: 1.0

Audio Sample: [lr_1.0](/training_audio/sgd_1.0.wav)  
Here you'll see how it diverges the neural network and it doen't converge after that as the gradients explode.

### LR: 1.0 with batch size 256

Audio Sample: [lr_1.0_bs_256](/training_audio/sgd_1.0_bs_256.wav)  
Warning: this gave me a minor headache.  
Notice how the batch size increase affects the gradients as they explode instantly and lead to the annoying low pitch thumping sound caused by gradients becoming "NaN"


## Code Brief
### Requirements
Requirements are listed in `requirements.txt` file. The project requires PyTorch along with Torchvision and PyAudio.  
Requirements can be installed as `pip install -r requirements.txt`

### Files
1. `model.py`: Model architecture file. By default takes batch size as 20. can be modified a little to generate 256 batch size result as well.  
2. `utils.py`: Utility functions for opening audio stream and other things.
3. `main.py`: Main file to run the code. By default generates the three LR based audios for batch size 20.  
Run the code as `python main.py`

## Conclusion
There is nothing of value here, just did this because it seemed like fun. I don't see anyone debugging their training process by analyzing weights using sound over visualizing weights using graphs.

