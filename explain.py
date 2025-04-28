# Core libraries
import torch

# Import the model definition
from allocator import AllocatorNN

# Load the trained model
model_path = 'besttt.pt'  # Update with your actual model path
model = AllocatorNN()

# Load the saved weights into the model
model.load_state_dict(torch.load(model_path))

# Set the model in evaluation mode
model.eval()

# If using GPU (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Now the model is ready to be used for inference