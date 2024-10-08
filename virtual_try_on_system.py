import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Define the generator network
class VTONGenerator(nn.Module):
    def __init__(self):
        super(VTONGenerator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()

        )

    def forward(self, person, clothes):
        x = torch.cat([person, clothes], dim=1)
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class VTONDataset(Dataset):
    def __init__(self, person_images, cloth_images, transform=None):
        self.person_images = person_images
        self.cloth_images = cloth_images
        self.transform = transform

    def __len__(self):
        return len(self.person_images)

    def __getitem__(self, idx):
        person_img = Image.open(self.person_images[idx]).convert('RGB')
        cloth_img = Image.open(self.cloth_images[idx]).convert('RGB')

        if self.transform:
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)

        return person_img, cloth_img

# Training function
def train_vton(model, dataloader, num_epochs, device):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for person, clothes in dataloader:
            person, clothes = person.to(device), clothes.to(device)

            optimizer.zero_grad()
            output = model(person, clothes)
            loss = criterion(output, person)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

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

    dataset = VTONDataset(['/content/03615_00.jpg', '/content/00891_00.jpg'], ['/content/06802_00.jpg', '/content/06429_00.jpg'], transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the model
    train_vton(model, dataloader, num_epochs=100, device=device)

    # Save the model
    torch.save(model.state_dict(), 'vton_model.pth')

if __name__ == "__main__":
    main()

