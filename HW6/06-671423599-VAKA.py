# put your image generator here

# Generate 9 random latent vectors
random_encoded_vectors = torch.randn(9, d).to(device)  # Generate random vectors

# Decode the random vectors
decoder.eval()  # Set the decoder to evaluation mode
with torch.no_grad():
    generated_images = decoder(random_encoded_vectors)

# Convert the output tensors to image format and plot
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    img = generated_images[i].cpu().detach().numpy()  # Convert to numpy array
    img = np.squeeze(img)  # Remove unnecessary dimensions
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.show()

from sklearn.metrics import accuracy_score
from scipy.stats import mode

encoded_images = []
true_labels = []
encoder.eval()
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        encoded = encoder(images).cpu().numpy()
        encoded_images.extend(encoded)
        true_labels.extend(labels.numpy())

encoded_images = np.array(encoded_images)
true_labels = np.array(true_labels)
best_kmeans = None
best_inertia = float('inf')

for i in range(10):  # For example, 10 different initializations
    kmeans = KMeans(n_clusters=10, n_init=10, random_state=i)
    kmeans.fit(encoded_images)

    if kmeans.inertia_ < best_inertia:
        best_kmeans = kmeans
        best_inertia = kmeans.inertia_

kmeans_labels = best_kmeans.labels_


# Step 3: Find the best mapping between k-means assignments and true labelse

def find_best_mapping(kmeans_labels, true_labels):
    label_mapping = {}
    for i in range(10):  # Assuming 10 clusters/digits
        mask = (kmeans_labels == i)
        # Find the most common true label in this cluster
        print(mask)
        print(true_labels[mask])
        mode_result = mode(true_labels[mask])
        print(mode_result)
        # mode_result.mode might be a scalar or an array
        if np.isscalar(mode_result.mode):
            most_common_label = mode_result.mode
        else:
            most_common_label = mode_result.mode[0]

        label_mapping[i] = most_common_label
    return label_mapping


label_mapping = find_best_mapping(kmeans_labels, true_labels)
print(label_mapping)
mapped_labels = [label_mapping[label] for label in kmeans_labels]
#print(mapped_labels)
# Step 4: Calculate and report accuracy
accuracy = accuracy_score(true_labels, mapped_labels)
print(f'Clustering accuracy: {accuracy}')
print(kmeans_labels)