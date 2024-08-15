import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to denormalize the tensor and convert to image
def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 0.5 + 0.5) * 255
    return image.astype(np.uint8)

# Function to display images
def show_images(person, clothes, output):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(tensor_to_image(person))
    ax1.set_title('Person')
    ax1.axis('off')

    ax2.imshow(tensor_to_image(clothes))
    ax2.set_title('Clothes')
    ax2.axis('off')

    ax3.imshow(tensor_to_image(output))
    ax3.set_title('Output')
    ax3.axis('off')

    plt.show()

# Modified training function to show output
def train_vton(model, dataloader, num_epochs, device):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for i, (person, clothes) in enumerate(dataloader):
            person, clothes = person.to(device), clothes.to(device)

            optimizer.zero_grad()
            output = model(person, clothes)
            loss = criterion(output, person)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:  # Show output every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                show_images(person[0], clothes[0], output[0])

# Function to test the model
def test_vton(model, person_image_path, cloth_image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    person_img = Image.open('/content/images (8).jpeg').convert('RGB')
    cloth_img = Image.open('/content/download (3).jpeg').convert('RGB')

    person_tensor = transform(person_img).unsqueeze(0).to(device)
    cloth_tensor = transform(cloth_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(person_tensor, cloth_tensor)

    show_images(person_tensor, cloth_tensor, output)

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = VTONGenerator().to(device)

    # Prepare the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = VTONDataset(['/content/images (8).jpeg', '/content/00891_00.jpg'], ['/content/download (3).jpeg', '/content/06429_00.jpg'], transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the model
    train_vton(model, dataloader, num_epochs=10, device=device)  # Reduced epochs for demonstration

    # Save the model
    torch.save(model.state_dict(), 'vton_model.pth')

    # Test the model
    test_vton(model, '/content/07573_00.jpg', '/content/07429_00.jpg', device)

if __name__ == "__main__":
    main()