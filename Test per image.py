import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
!pip install timm
import timm
classes= {'Bear': 0, 'Brown bear': 1, 'Bull': 2, 'Butterfly': 3, 'Camel': 4, 'Canary': 5, 'Caterpillar': 6, 'Cattle': 7, 'Centipede': 8, 'Cheetah': 9, 'Chicken': 10, 'Crab': 11, 'Crocodile': 12, 'Deer': 13, 'Duck': 14, 'Eagle': 15, 'Elephant': 16, 'Fish': 17, 'Fox': 18, 'Frog': 19, 'Giraffe': 20, 'Goat': 21, 'Goldfish': 22, 'Goose': 23, 'Hamster': 24, 'Harbor seal': 25, 'Hedgehog': 26, 'Hippopotamus': 27, 'Horse': 28, 'Jaguar': 29, 'Jellyfish': 30, 'Kangaroo': 31, 'Koala': 32, 'Ladybug': 33, 'Leopard': 34, 'Lion': 35, 'Lizard': 36, 'Lynx': 37, 'Magpie': 38, 'Monkey': 39, 'Moths and butterflies': 40, 'Mouse': 41, 'Mule': 42, 'Ostrich': 43, 'Otter': 44, 'Owl': 45, 'Panda': 46, 'Parrot': 47, 'Penguin': 48, 'Pig': 49, 'Polar bear': 50, 'Rabbit': 51, 'Raccoon': 52, 'Raven': 53, 'Red panda': 54, 'Rhinoceros': 55, 'Scorpion': 56, 'Sea lion': 57, 'Sea turtle': 58, 'Seahorse': 59, 'Shark': 60, 'Sheep': 61, 'Shrimp': 62, 'Snail': 63, 'Snake': 64, 'Sparrow': 65, 'Spider': 66, 'Squid': 67, 'Squirrel': 68, 'Starfish': 69, 'Swan': 70, 'Tick': 71, 'Tiger': 72, 'Tortoise': 73, 'Turkey': 74, 'Turtle': 75, 'Whale': 76, 'Woodpecker': 77, 'Worm': 78, 'Zebra': 79}

# Load the pre-trained model
model = torch.load('/content/animals_best_model.pth')
# Load the model onto the GPU
m1 = timm.create_model("rexnet_150", pretrained = False, num_classes = len(classes)) # Create a new model instance with the same architecture but not pretrained
m1.load_state_dict(model) #Load the saved weights
m1.eval()
m1.to("cuda") #Move the model to the GPU


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization parameters (mean)
                         std=[0.229, 0.224, 0.225])    # Normalization parameters (std)
])

# Function to process the image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add a batch dimension
    # Move the image to the GPU
    image = image.to("cuda")
    return image

# Prompt the user for an image path
image_path = input("Enter the path of the file:")

# Process the input image
input_tensor = process_image(image_path)

# Run the model on the input tensor
with torch.no_grad():
    output = m1(input_tensor)

# Obtain the predicted class
_, predicted = torch.max(output, 1)
class_names=list(classes)
predicted_class_name = class_names[predicted.item()]
print(f"Predicted class: {predicted.item()}")
print("The animal is :",predicted_class_name)
